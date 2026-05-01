'''
Simamba model implementation for vLLM.
Defines:
    - Decoder layer using SimambaMixer (SSM-based attention-free block)
    - Backbone model with pipeline parallelism support
    - Causal LM wrapper with weight loading + logits computation
'''

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateDtypeCalculator,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.mamba.simamba_mixer import (
    SIMAMBA_BACKEND_TRITON,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors

from .utils import AutoWeightsLoader, make_empty_intermediate_tensors_factory, make_layers, maybe_prefix
from ..layers.mamba.simamba_mixer import SimambaMixer
from vllm.distributed import divide


def _cfg_get(config: object, name: str, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _get_ssm_cfg(config: object) -> dict:
    ssm_cfg = _cfg_get(config, "ssm_cfg", {}) or {}
    return dict(ssm_cfg)


def _get_hidden_size(config: object) -> int:
    return _cfg_get(config, "hidden_size", _cfg_get(config, "d_model"))


def _get_num_layers(config: object) -> int:
    return _cfg_get(config, "num_hidden_layers", _cfg_get(config, "n_layer"))


def _get_norm_eps(config: object) -> float:
    return _cfg_get(config, "layer_norm_epsilon", 1e-5)


# Decoder layer
class SimambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: object,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # unsued for now
        del quant_config
        super().__init__()
        ssm_cfg = _get_ssm_cfg(config)
        hidden_size = _get_hidden_size(config)

        # Core mixer (SSM block replacing attention)
        self.mixer = SimambaMixer(
            hidden_size=hidden_size,
            ssm_state_size=ssm_cfg.get("d_state", _cfg_get(config, "state_size", 128)),
            expand=ssm_cfg.get("expand", _cfg_get(config, "expand", 2)),
            head_dim=ssm_cfg.get("headdim", _cfg_get(config, "head_dim", 64)),
            n_groups=ssm_cfg.get("ngroups", _cfg_get(config, "n_groups", 1)),
            rope_fraction=ssm_cfg.get("rope_fraction", 0.5),
            use_midpoint_control=ssm_cfg.get("use_midpoint_control", False),
            simamba_backend=ssm_cfg.get("simamba_backend", SIMAMBA_BACKEND_TRITON),
            simpson_boundary_mode=ssm_cfg.get("simpson_boundary_mode", "zero_pad"),
            is_outproj_norm=ssm_cfg.get("is_outproj_norm", False),
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.mixer",
        )
        # Pre-norm (RMSNorm)
        self.norm = RMSNorm(hidden_size, eps=_get_norm_eps(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        # Pre-norm + residual handling
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            # Fused norm with residual update
            hidden_states, residual = self.norm(hidden_states, residual)

        output = torch.empty_like(hidden_states)

        # Run SSM mixer
        self.mixer(hidden_states, output)
        return output, residual


# explicit dynamic_arg_dims
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "inputs_embeds": {0: "batch_size", 1: "seq_len"},
        "positions": {0: "batch_size", 1: "seq_len"},
    }
)
class SimambaModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        hidden_size = _get_hidden_size(config)
        vocab_size = _cfg_get(config, "vocab_size")

        self.embeddings = VocabParallelEmbedding(vocab_size, hidden_size)
        # Build decoder layers with pipeline parallel partitioning
        self.start_layer, self.end_layer, self.layers = make_layers(
            _get_num_layers(config),
            lambda layer_prefix: SimambaDecoderLayer(
                config,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=layer_prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        self.norm_f = RMSNorm(hidden_size, eps=_get_norm_eps(config))

        # Factory for pipeline intermediate tensors
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Positions unused but kept for API compatibility
        del positions

        # Pipeline parallel handling
        if get_pp_group().is_first_rank:
            # First stage: create embeddings
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            # Later stages: receive tensors from previous stage
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Run assigned layer slice
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(hidden_states=hidden_states, residual=residual)

        # If not last stage then can pass tensors forward
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        # Final norm at last stage
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states


class SimambaForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP):
    '''
    Causal language model wrapper:
        - Adds LM head
        - Handles logits computation
        - Provides state metadata for caching
    '''
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        # Compute dtype for each cached state tensor
        kv_dtype, temporal_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )
        return (
            temporal_dtype,
            temporal_dtype,
            kv_dtype,
            kv_dtype,
            kv_dtype,
            kv_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        '''
        Compute shape of each cached state tensor
        '''
        hf_config = vllm_config.model_config.hf_config
        ssm_cfg = _get_ssm_cfg(hf_config)

        hidden_size = _get_hidden_size(hf_config)
        expand = ssm_cfg.get("expand", _cfg_get(hf_config, "expand", 2))
        head_dim = ssm_cfg.get("headdim", _cfg_get(hf_config, "head_dim", 64))
        d_state = ssm_cfg.get("d_state", _cfg_get(hf_config, "state_size", 128))
        rope_fraction = ssm_cfg.get("rope_fraction", 0.5)

        # Derived dimensions
        d_inner = hidden_size * expand
        num_heads = d_inner // head_dim

        # Adjust for tensor parallelism
        tp = vllm_config.parallel_config.tensor_parallel_size
        local_heads = divide(num_heads, tp)

        # Compute number of rotary angles
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        num_rope_angles = split_tensor_size // 2
        return (
            (local_heads, num_rope_angles),
            (local_heads, head_dim, d_state),
            (local_heads, d_state),
            (local_heads, d_state),
            (local_heads, head_dim),
            (local_heads, head_dim),
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        # All states use temporal copy behavior
        return (
            get_temporal_copy_spec,
            get_temporal_copy_spec,
            get_temporal_copy_spec,
            get_temporal_copy_spec,
            get_temporal_copy_spec,
            get_temporal_copy_spec,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

        # Backbone model
        self.backbone = SimambaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        vocab_size = _cfg_get(self.config, "vocab_size")
        hidden_size = _get_hidden_size(self.config)

        # Weight tying (embedding = LM head)
        if _cfg_get(self.config, "tie_word_embeddings", _cfg_get(self.config, "tie_embeddings", True)):
            self.lm_head = self.backbone.embeddings
        else:
            self.lm_head = ParallelLMHead(
                vocab_size,
                hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        # Logits post-processing (e.g., scaling, masking)
        self.logits_processor = LogitsProcessor(vocab_size)
        self.make_empty_intermediate_tensors = self.backbone.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Apply LM head + logits processing
        return self.backbone.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        del kwargs
        return self.backbone(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        '''
        Custom weight loading:
        - Handles fused projections (splits into multiple params)
        - Supports embedding tying fallback
        - Uses custom weight_loader hooks when available
        '''
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        def maybe_remap_name(name: str) -> str:
            # Handle naming differences across checkpoints
            if name.startswith("backbone.embedding."):
                return name.replace("backbone.embedding.", "backbone.embeddings.", 1)
            if name == "lm_head.weight" and "lm_head.weight" not in params_dict:
                return "backbone.embeddings.weight"
            return name

        for name, loaded_weight in weights:
            name = maybe_remap_name(name)

            # Handle fused input projection weights
            if name.endswith(".mixer.in_proj.weight"):
                prefix = name[: -len("in_proj.weight")]

                # Target split projections
                targets = [
                    f"{prefix}z_proj.weight",
                    f"{prefix}x_proj.weight",
                    f"{prefix}b_proj.weight",
                    f"{prefix}c_proj.weight",
                    f"{prefix}dt_proj.weight",
                    f"{prefix}a_proj.weight",
                    f"{prefix}simpson_proj.weight",
                    f"{prefix}angle_proj.weight",
                ]
                midpoint_name = f"{prefix}midpoint_proj.weight"
                if midpoint_name in params_dict:
                    targets.insert(-1, midpoint_name)

                # Load into each split parameter
                for target in targets:
                    if target not in params_dict:
                        continue
                    param = params_dict[target]
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight)
                    loaded.add(target)
                continue

            # Just skip missing biases
            # NOTE: come back and check later if there's a better alternative
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", None)

            # Use custom loader
            if weight_loader is None:
                param.data.copy_(loaded_weight)
            else:
                weight_loader(param, loaded_weight)
            loaded.add(name)

        return loaded
