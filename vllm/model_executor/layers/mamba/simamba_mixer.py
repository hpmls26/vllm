'''
Simamba sequence mixing layer.
This module implements a custom Mamba-like state space model (SSM) layer
used inside vLLM for fast inference.
    - The continuous-time dynamics are discretized using Simpson’s rule
    - Supports fast inference (decode) and batched prefill
    - Integrates with vLLM’s KV/state cache system
'''

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNorm, RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from vllm.model_executor.layers.mamba.ops.simamba import (
    SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
    SIMAMBA_SUPPORTED_BOUNDARY_MODES,
    simamba_siso_combined,
    simamba_siso_step,
    simamba_triton_siso_combined,
    simamba_triton_siso_step,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


SIMAMBA_BACKEND_REFERENCE = "reference"
SIMAMBA_BACKEND_TRITON = "triton"


def _cfg_get(config: object, name: str, default=None):
    '''
    Extract attribute from config object or dict.
    Used for compatibility across config formats.
    '''
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _split_lengths(query_start_loc: torch.Tensor | None, fallback_len: int) -> list[int]:
    '''
    Convert prefix sum style sequence boundaries into per sequence lengths.
    Used during decode to split batched queries into individual requests.
    '''
    if query_start_loc is None:
        return [1] * fallback_len
    return torch.diff(query_start_loc).tolist()


@PluggableLayer.register("simamba_mixer")
class SimambaMixer(MambaBase, PluggableLayer):
    '''
     Main Simamba mixing layer.
        - Project hidden states into SSM parameter space
        - Run prefill (batched sequence processing)
        - Run decode (incremental token processing)
        - Maintain and update per-request state cache
        - Support tensor parallel execution
    '''
    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        expand: int,
        head_dim: int,
        n_groups: int,
        rope_fraction: float,
        use_midpoint_control: bool,
        simamba_backend: str,
        simpson_boundary_mode: str,
        is_outproj_norm: bool,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Tensor-parallel configuration
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        # Core model dimensions
        self.hidden_size = hidden_size
        self.ssm_state_size = ssm_state_size
        self.expand = expand

        # Head structure
        self.head_dim = head_dim
        self.n_groups = n_groups

        # Rotary embedding configuration
        self.rope_fraction = rope_fraction

        # Algorithm adjustments
        self.use_midpoint_control = use_midpoint_control
        self.is_outproj_norm = is_outproj_norm

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        if simamba_backend not in (
            SIMAMBA_BACKEND_REFERENCE,
            SIMAMBA_BACKEND_TRITON,
        ):
            raise ValueError(f"Unsupported simamba backend: {simamba_backend}")
        if simpson_boundary_mode not in SIMAMBA_SUPPORTED_BOUNDARY_MODES:
            raise ValueError(
                f"Unsupported simpson boundary mode: {simpson_boundary_mode}"
            )
        self.simamba_backend = simamba_backend
        self.simpson_boundary_mode = simpson_boundary_mode
        
        self.d_inner = hidden_size * expand

        # Ensure valid head partitioning
        assert self.d_inner % head_dim == 0, (
            f"d_inner ({self.d_inner}) must divide head_dim ({head_dim})"
        )
        self.num_heads = self.d_inner // head_dim

        # Tensor parallel split across heads
        assert self.num_heads % self.tp_size == 0, (
            f"num_heads ({self.num_heads}) must divide tp_size ({self.tp_size})"
        )
        self.local_num_heads = self.num_heads // self.tp_size
        self.local_d_inner = self.d_inner // self.tp_size
        self.local_head_controls = self.local_num_heads

        # Constraint: grouped states not yet supported with TP
        if n_groups != 1 and self.tp_size != 1:
            raise NotImplementedError(
                "Simamba tensor parallel currently supports n_groups=1 only."
            )

        # Validate rotary embedding structure
        if rope_fraction not in (0.5, 1.0):
            raise ValueError(f"Unsupported rope_fraction={rope_fraction}")

        # Compute rotary dimension split
        split_tensor_size = int(ssm_state_size * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2

        # Ensure valid config
        if self.num_rope_angles <= 0:
            raise ValueError("Simamba requires at least one rotary angle pair.")

        num_head_controls = 4 if use_midpoint_control else 3


        # Input projection layers
        self.z_proj = ColumnParallelLinear(
            hidden_size,
            self.d_inner,
            bias=False,
            prefix=f"{prefix}.z_proj",
        )
        self.x_proj = ColumnParallelLinear(
            hidden_size,
            self.d_inner,
            bias=False,
            prefix=f"{prefix}.x_proj",
        )
        self.b_proj = ReplicatedLinear(
            hidden_size,
            n_groups * ssm_state_size,
            bias=False,
            prefix=f"{prefix}.b_proj",
        )
        self.c_proj = ReplicatedLinear(
            hidden_size,
            n_groups * ssm_state_size,
            bias=False,
            prefix=f"{prefix}.c_proj",
        )
        self.dt_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads,
            bias=False,
            prefix=f"{prefix}.dt_proj",
        )
        self.a_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads,
            bias=False,
            prefix=f"{prefix}.a_proj",
        )
        self.simpson_proj = ColumnParallelLinear(
            hidden_size,
            self.num_heads,
            bias=False,
            prefix=f"{prefix}.simpson_proj",
        )
        self.midpoint_proj = (
            ColumnParallelLinear(
                hidden_size,
                self.num_heads,
                bias=False,
                prefix=f"{prefix}.midpoint_proj",
            )
            if use_midpoint_control
            else None
        )
        self.angle_proj = ReplicatedLinear(
            hidden_size,
            self.num_rope_angles,
            bias=False,
            prefix=f"{prefix}.angle_proj",
        )

        # Learnable parameters
        # Discretization bias for time step
        self.dt_bias = nn.Parameter(torch.empty(self.local_num_heads, dtype=torch.float32))

        # State-space biases (per-head)
        self.B_bias = nn.Parameter(
            torch.ones(self.local_num_heads, self.ssm_state_size, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.local_num_heads, self.ssm_state_size, dtype=torch.float32)
        )

        # Residual scaling
        self.D = nn.Parameter(torch.ones(self.local_num_heads, dtype=torch.float32))

        # Normalization modules for stability
        self.B_norm = RMSNorm(self.ssm_state_size, eps=1e-5)
        self.C_norm = RMSNorm(self.ssm_state_size, eps=1e-5)
        self.norm = (
            RMSNormGated(
                self.local_d_inner,
                eps=1e-5,
                group_size=self.head_dim,
                norm_before_gate=True,
            )
            if is_outproj_norm
            else None
        )
        self.out_proj = RowParallelLinear(
            self.d_inner,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            prefix=f"{prefix}.out_proj",
        )

        # Initialize parameters + register custom loaders
        self._init_dt_bias()
        self._set_weight_loaders()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Allocate per-layer state cache (6 tensors total)
        self.kv_cache = tuple(torch.tensor([]) for _ in range(6))

    def _init_dt_bias(self) -> None:
        '''
        Initialize learned time-step bias for SSM discretization.

        This biases the softplus-transformed time constant so that values are positive and well-scaled
        and initial dynamics are stable across heads.
        '''
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4

        # Sample log-uniform distribution for time-step initialization
        dt = torch.exp(
            torch.rand(self.local_num_heads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )

        # Ensure numerical stability: avoid extremely small steps
        dt = torch.clamp(dt, min=dt_init_floor)

        # Inverse softplus transform so that after softplus we recover dt
        dt_bias = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_bias.copy_(dt_bias)

    def _set_weight_loaders(self) -> None:
        '''
        Register custom weight-loading logic for tensor-parallel and grouped parameters.

        This function defines how pretrained weights are:
            - sliced per tensor-parallel rank
            - mapped into internal parameter layout
            - loaded for both dense and grouped SSM components
        '''
        local_head_start = self.tp_rank * self.local_num_heads

        # Global parameter layout offsets inside flattened checkpoint tensors
        global_offsets = {
            "z": 0,
            "x": self.d_inner,
            "b": 2 * self.d_inner,
            "c": 2 * self.d_inner + self.n_groups * self.ssm_state_size,
            "dt": 2 * self.d_inner + 2 * self.n_groups * self.ssm_state_size,
            "a": 2 * self.d_inner + 2 * self.n_groups * self.ssm_state_size + self.num_heads,
            "simpson": 2 * self.d_inner
            + 2 * self.n_groups * self.ssm_state_size
            + 2 * self.num_heads,
        }
        if self.use_midpoint_control:
            global_offsets["midpoint"] = (
                2 * self.d_inner
                + 2 * self.n_groups * self.ssm_state_size
                + 3 * self.num_heads
            )
            global_offsets["angles"] = (
                2 * self.d_inner
                + 2 * self.n_groups * self.ssm_state_size
                + 4 * self.num_heads
            )
        else:
            global_offsets["angles"] = (
                2 * self.d_inner
                + 2 * self.n_groups * self.ssm_state_size
                + 3 * self.num_heads
            )

        def make_in_proj_loader(name: str, global_size: int, sharded: bool):
            '''
            Factory for parameter loader functions.

            Handles slicing global checkpoint tensor into component blocks, tensor 
            parallel sharding across heads, and correct offset selection per parameter type
            '''
            global_start = global_offsets[name]

            def loader(param: Parameter, loaded_weight: torch.Tensor) -> None:
                # Extract relevant slice from flattened checkpoint tensor
                shard = loaded_weight[global_start : global_start + global_size]
                if sharded:
                    if global_size == self.num_heads:
                        start = local_head_start
                    else:
                        start = self.tp_rank * param.shape[0]
                    shard = shard[start : start + param.shape[0]]
                param.data.copy_(shard)

            return loader

         # Register loaders for all projection layers
        set_weight_attrs(
            self.z_proj.weight,
            {"weight_loader": make_in_proj_loader("z", self.d_inner, True)},
        )
        set_weight_attrs(
            self.x_proj.weight,
            {"weight_loader": make_in_proj_loader("x", self.d_inner, True)},
        )
        set_weight_attrs(
            self.b_proj.weight,
            {"weight_loader": make_in_proj_loader("b", self.n_groups * self.ssm_state_size, False)},
        )
        set_weight_attrs(
            self.c_proj.weight,
            {"weight_loader": make_in_proj_loader("c", self.n_groups * self.ssm_state_size, False)},
        )
        set_weight_attrs(
            self.dt_proj.weight,
            {"weight_loader": make_in_proj_loader("dt", self.num_heads, True)},
        )
        set_weight_attrs(
            self.a_proj.weight,
            {"weight_loader": make_in_proj_loader("a", self.num_heads, True)},
        )
        set_weight_attrs(
            self.simpson_proj.weight,
            {"weight_loader": make_in_proj_loader("simpson", self.num_heads, True)},
        )
        if self.midpoint_proj is not None:
            set_weight_attrs(
                self.midpoint_proj.weight,
                {
                    "weight_loader": make_in_proj_loader(
                        "midpoint", self.num_heads, True
                    )
                },
            )
        set_weight_attrs(
            self.angle_proj.weight,
            {
                "weight_loader": make_in_proj_loader(
                    "angles", self.num_rope_angles, False
                )
            },
        )

        def local_head_loader(param: Parameter, loaded_weight: torch.Tensor) -> None:
            # Load only this tensor-parallel shard of head-wise parameters
            param.data.copy_(
                loaded_weight[
                    local_head_start : local_head_start + param.shape[0]
                ]
            )

        def local_head_state_loader(param: Parameter, loaded_weight: torch.Tensor) -> None:
            # Same logic but used for state-related bias tensors
            param.data.copy_(
                loaded_weight[
                    local_head_start : local_head_start + param.shape[0]
                ]
            )

        set_weight_attrs(self.dt_bias, {"weight_loader": local_head_loader})
        set_weight_attrs(self.D, {"weight_loader": local_head_loader})
        set_weight_attrs(self.B_bias, {"weight_loader": local_head_state_loader})
        set_weight_attrs(self.C_bias, {"weight_loader": local_head_state_loader})
        if self.norm is not None:
            start = self.tp_rank * self.local_d_inner

            def norm_loader(param: Parameter, loaded_weight: torch.Tensor) -> None:
                param.data.copy_(loaded_weight[start : start + param.shape[0]])

            set_weight_attrs(self.norm.weight, {"weight_loader": norm_loader})

    @property
    def chunk_size(self) -> int:
        '''
        Chunk size used for sequence processing during prefill.
        Controls how long sequences are split when using SSM scan kernels.
        '''
        assert self.model_config is not None
        return self.model_config.get_mamba_chunk_size()

    def _project(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor | None]:
        '''
        Project hidden states into all SSM parameter streams.

        Input transformation step:
        hidden_states -> (z, x, b, c, dt, a, simpson, angles, optional midpoint)
        '''
        z = self.z_proj(hidden_states)[0]
        x = self.x_proj(hidden_states)[0]
        b = self.b_proj(hidden_states)[0]
        c = self.c_proj(hidden_states)[0]
        dd_dt = self.dt_proj(hidden_states)[0]
        dd_a = self.a_proj(hidden_states)[0]
        simpson = self.simpson_proj(hidden_states)[0]
        midpoint = self.midpoint_proj(hidden_states)[0] if self.midpoint_proj else None
        angles = self.angle_proj(hidden_states)[0]
        return {
            "z": z,
            "x": x,
            "b": b,
            "c": c,
            "dd_dt": dd_dt,
            "dd_a": dd_a,
            "simpson": simpson,
            "midpoint": midpoint,
            "angles": angles,
        }

    def _prep_sequence_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        '''
        Convert raw projections into structured SSM inputs.
        Returns tensors ready for prefill kernel execution.
        '''
        p = self._project(hidden_states)
        n = hidden_states.shape[0]

        z = p["z"].view(n, self.local_num_heads, self.head_dim)
        x = p["x"].view(n, self.local_num_heads, self.head_dim)

        b = self.B_norm(p["b"].view(n, self.n_groups, self.ssm_state_size))
        c = self.C_norm(p["c"].view(n, self.n_groups, self.ssm_state_size))

        dd_dt = p["dd_dt"]
        dd_a = p["dd_a"]

        simpson = torch.sigmoid(p["simpson"])
        midpoint = torch.sigmoid(p["midpoint"]) if p["midpoint"] is not None else None

        # Stable SSM discretization
        dt = F.softplus(dd_dt + self.dt_bias)
        a = -F.softplus(dd_a.float())
        adt = a * dt
        return z, x, b, c, adt, midpoint, dt, p["angles"], simpson

    def _scatter_states(
        self,
        state_indices: torch.Tensor,
        states: Sequence[torch.Tensor],
    ) -> None:
        '''
        Write updated SSM states back into the global KV/state cache.

        Only valid (non-NULL) entries are updated.
        '''
        valid = state_indices != NULL_BLOCK_ID
        if not torch.any(valid):
            return

        indices = state_indices[valid].long()
        for cache_state, src in zip(self.kv_cache, states):
            cache_state[indices] = src[valid].to(cache_state.dtype)

    def _gather_states(
        self,
        state_indices: torch.Tensor,
        *,
        state_cols: torch.Tensor | None = None,
        has_initial: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        '''
        Fetch cached SSM states for given request indices.
        Handles empty cache initialization (zeros), partial state selection (state_cols),
        and filtering invalid entries (NULL_BLOCK_ID)
        '''
        batch = state_indices.shape[0]

        if state_cols is None:
            indices = state_indices
        else:
            row_ids = torch.arange(batch, device=state_indices.device)
            indices = state_indices[row_ids, state_cols]

        angle_shape, ssm_shape, k_shape, _, v_shape, _ = self.get_state_shape()
        dtypes = self.get_state_dtype()

        # Initialize all states as zeros (default when cache is empty or invalid)
        zeros = (
            torch.zeros((batch,) + angle_shape, device=state_indices.device, dtype=dtypes[0]),
            torch.zeros((batch,) + ssm_shape, device=state_indices.device, dtype=dtypes[1]),
            torch.zeros((batch,) + k_shape, device=state_indices.device, dtype=dtypes[2]),
            torch.zeros((batch,) + k_shape, device=state_indices.device, dtype=dtypes[3]),
            torch.zeros((batch,) + v_shape, device=state_indices.device, dtype=dtypes[4]),
            torch.zeros((batch,) + v_shape, device=state_indices.device, dtype=dtypes[5]),
        )

        valid = indices != NULL_BLOCK_ID
        if has_initial is not None:
            valid = valid & has_initial

        # If nothing is valid return all zero states immediately
        if not torch.any(valid):
            return zeros
        
        # Clone zero tensors to fill only valid positions
        out = [state.clone() for state in zeros]
        
        # Gather valid cache indices
        src_indices = indices[valid].long()
        # Copy cached states into output tensors at valid positions
        for dst_state, cache_state in zip(out, self.kv_cache):
            dst_state[valid] = cache_state[src_indices].to(dst_state.dtype)
        return tuple(out)

    def _run_prefill(
        self,
        hidden_states: torch.Tensor,
        metadata: Mamba2AttentionMetadata,
        p: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        '''
        Prefill processes full sequences (no autoregressive stepping)
        '''

        assert metadata.query_start_loc_p is not None
        assert metadata.state_indices_tensor_p is not None

        # Load initial states from cache (or zeros if missing)
        state_indices = metadata.state_indices_tensor_p
        init_states = self._gather_states(
            state_indices,
            has_initial=metadata.has_initial_states_p,
        )
        
        # Switch: use precomputed projections
        n = hidden_states.shape[0]
        z = p["z"].view(n, self.local_num_heads, self.head_dim)   # shaped here, once
        x = p["x"].view(n, self.local_num_heads, self.head_dim)
        b = self.B_norm(p["b"].view(n, self.n_groups, self.ssm_state_size))
        c = self.C_norm(p["c"].view(n, self.n_groups, self.ssm_state_size))
        dt = F.softplus(p["dd_dt"] + self.dt_bias)
        a = -F.softplus(p["dd_a"].float())
        adt = a * dt
        simpson = torch.sigmoid(p["simpson"])
        midpoint = torch.sigmoid(p["midpoint"]) if p["midpoint"] is not None else None

        # Run full sequence SSM computation (either reference or Triton backend)
        if self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
            out, *final_states = simamba_siso_combined(
                Q=q,
                K=k,
                V=v,
                ADT=adt_seq,
                DT=dt_seq,
                Simpson=simpson_seq,
                Midpoint=midpoint_seq,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                Angles=angles_seq,
                D=self.D,
                Z=z_seq if not self.is_outproj_norm else None,
                Input_States=init_states,
                chunk_size=self.chunk_size,
                return_final_states=True,
                boundary_mode=self.simpson_boundary_mode,
            )
        else:
            out, *final_states = simamba_triton_siso_combined(
                Q=q,
                K=k,
                V=v,
                ADT=adt_seq,
                DT=dt_seq,
                Simpson=simpson_seq,
                Midpoint=midpoint_seq,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                Angles=angles_seq,
                D=self.D,
                Z=z_seq if not self.is_outproj_norm else None,
                Initial_States=init_states,
                chunk_size=self.chunk_size,
                return_final_states=True,
                cu_seqlens=metadata.query_start_loc_p,
            )
        final_states_t = tuple(final_states)
        self._scatter_states(state_indices, final_states_t)

        # Flatten back to [tokens, dim]
        # NOTE: we now return z alongisde in its original form for caching
        return out.view(n, self.local_d_inner), z

    def _run_decode(
        self,
        hidden_states: torch.Tensor,
        metadata: Mamba2AttentionMetadata,
        p: dict[str, torch.Tensor | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Decode handles autoregressive generation (step-by-step or short chunks)
        '''
        assert metadata.state_indices_tensor_d is not None
        state_indices = metadata.state_indices_tensor_d

        # Split decode tokens into per-request lengths
        q_lens = _split_lengths(metadata.query_start_loc_d, metadata.num_decodes)
        out_chunks: list[torch.Tensor] = []

        # Determine which column of state_indices to read (last accepted token)
        start_cols = (
            metadata.num_accepted_tokens.to(torch.int64) - 1
            if metadata.num_accepted_tokens is not None
            else torch.zeros(metadata.num_decodes, device=hidden_states.device, dtype=torch.int64)
        )

        # Fast path: all sequences are single-token -> batch step
        if all(q_len == 1 for q_len in q_lens):
            init_states = self._gather_states(state_indices, state_cols=start_cols)
            y_flat, next_states = self._run_step_batch(hidden_states, init_states, p)
            self._scatter_states(state_indices[:, 0], next_states)
            return y_flat, p["z"]

        # General path: variable-length decode per request
        token_offset = 0
        for req_idx, q_len in enumerate(q_lens):
            states = self._gather_states(
                state_indices[req_idx : req_idx + 1],
                state_cols=start_cols[req_idx : req_idx + 1],
            )
            req_out: list[torch.Tensor] = []

            # Step through tokens sequentially
            for step_idx in range(q_len):
                tok = slice(token_offset, token_offset + 1)
                p_step = {k: (v[tok] if v is not None else None) for k, v in p.items()}
                # p_step passed not reprojected compared to old double projection method in decode path
                y_step, next_states = self._run_step_batch(
                    hidden_states[tok], states, p_step   
                )

                # Save updated state for this token
                self._scatter_states(
                    state_indices[req_idx : req_idx + 1, step_idx],
                    states,
                )
                req_out.append(out_step)
                token_offset += 1

            out_chunks.append(torch.cat(req_out, dim=0))

        return torch.cat(out_chunks, dim=0)

    def _run_step_batch(
        self,
        hidden_states: torch.Tensor,
        states: tuple[torch.Tensor, ...],
        p: dict[str, torch.Tensor | None]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        '''
        Single-step SSM update (used in decode)
        '''
        # Project inputs into SSM parameter space
        #p = self._project(hidden_states)

        n = hidden_states.shape[0]
        z = p["z"]   # already shaped [T, local_num_heads, head_dim] from _project
        x = p["x"]
        b = self.B_norm(p["b"].view(n, self.n_groups, self.ssm_state_size))
        c = self.C_norm(p["c"].view(n, self.n_groups, self.ssm_state_size))
        dt      = F.softplus(p["dd_dt"] + self.dt_bias)
        a       = -F.softplus(p["dd_a"].float())
        simpson = torch.sigmoid(p["simpson"])
        midpoint = torch.sigmoid(p["midpoint"]) if p["midpoint"] is not None else None
        angles  = p["angles"].unsqueeze(1).expand(-1, self.local_num_heads, -1)

        # Choose backend kernel
        op = (
            simamba_siso_step
            if self.simamba_backend == SIMAMBA_BACKEND_REFERENCE
            else simamba_triton_siso_step
        )

        # Execute one SSM step
        if self.simamba_backend == SIMAMBA_BACKEND_REFERENCE:
            out, next_states = op(
                Q=c,
                K=b,
                V=x,
                ADT=a * dt,
                DT=dt,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                Angles=angles,
                Input_States=states,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
                boundary_mode=self.simpson_boundary_mode,
            )
        else:
            out, next_states = op(
                Q=c,
                K=b,
                V=x,
                ADT=a * dt,
                DT=dt,
                Simpson=simpson,
                Midpoint=midpoint,
                Q_bias=self.C_bias,
                K_bias=self.B_bias,
                Angles=angles,
                Input_States=states,
                Output_States=states,
                D=self.D,
                Z=z if not self.is_outproj_norm else None,
            )
        out = out.view(-1, self.local_d_inner)
        return out, tuple(next_states)

    def _finalize(self, y: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        '''
        Apply optional output normalization (e.g. gated / RMSNorm variant)
        '''
        if self.norm is not None and z_local is not None:
            # Flatten now that we get original z
            z_flat = z.reshape(y.shape[0], self.local_d_inner)
            y = self.norm(y, z_flat)
        return self.out_proj(y)[0]

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        '''
        Entry point used by the runtime (calls custom fused op)
        Delegates execution to registered custom op for performance
        '''
        torch.ops.vllm.simamba_mixer(hidden_states, output, self.prefix)

    def forward_impl(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
        '''
        '''
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        # Case 1: No metadata so treat as single prefill (standalone forward)
        if attn_metadata is None:
            p = self._project(hidden_states)
            y_local, z_shaped = self._run_prefill(hidden_states, dummy_metadata, p)
            output[: hidden_states.shape[0]] = self._finalize(
                y_local, z_shaped if self.norm is not None else None
            )
            return
        
        # Case 2: Normal execution with scheduler metadata
        assert isinstance(attn_metadata, dict)
        metadata = attn_metadata[self.prefix]
        assert isinstance(metadata, Mamba2AttentionMetadata)

        num_actual_tokens = metadata.num_prefill_tokens + metadata.num_decode_tokens
        if num_actual_tokens == 0:
            return
        
        # Trim to only active tokens
        hidden_states = hidden_states[:num_actual_tokens]

        outputs: list[torch.Tensor] = []
        z_chunks: list[torch.Tensor] = []

        decode_tokens = metadata.num_decode_tokens
        prefill_tokens = metadata.num_prefill_tokens

        # Decode phase (autoregressive step updates)
        if decode_tokens > 0:
            hs_d = hidden_states[:decode_tokens]

            # Project once to extract z for normalization later
            p_d = self._project(hs_d)

            # Avoid reprojection
            y_d, z_d = self._run_decode(hs_d, metadata, p_d)
            # Step wise decode
            outputs.append(y_d)
            z_chunks.append(z_d)

        # Prefill phase (full sequence processing
        if prefill_tokens > 0:
            hs_p = hidden_states[decode_tokens:]
            p_p = self._project(hs_p)
            y_p, z_p = self._run_prefill(hs_p, metadata, p_p)  # p passed in to fix double projection
            outputs.append(y_p)
            z_chunks.append(z_p)

        # Concatenate outputs from decode + prefill paths
        y_local = torch.cat(outputs, dim=0)
        # Concatenate normalization inputs 
        z_local = torch.cat(z_chunks, dim=0) if self.norm is not None else None
        output[:num_actual_tokens] = self._finalize(y_local, z_local)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        '''
        Determine dtype for cached states:
            - kv_dtype: used for key/value-like tensors
            - temporal_dtype: used for time-dependent SSM states
        '''
        assert self.model_config is not None
        assert self.cache_config is not None

        kv_dtype, temporal_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

        # Return dtype per state component
        # Check if align with get_state_shape in test?
        return (
            temporal_dtype,
            temporal_dtype,
            kv_dtype,
            kv_dtype,
            kv_dtype,
            kv_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        '''
        Shapes for each cached state tensor 
        '''
        return (
            (self.local_num_heads, self.num_rope_angles),                 # Angles
            (self.local_num_heads, self.head_dim, self.ssm_state_size),   # SSM state
            (self.local_num_heads, self.ssm_state_size),                  # B / K
            (self.local_num_heads, self.ssm_state_size),                  # C / Q
            (self.local_num_heads, self.head_dim),                        # V
            (self.local_num_heads, self.head_dim),                        # Output / residual
        )

    @property
    def mamba_type(self) -> str:
        '''
        Identifier for mixer type just for framework 
        '''
        return "simamba"


def simamba_mixer(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    '''
    '''
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_impl(hidden_states, output)


def simamba_mixer_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    '''
    DO NOT DELTETE
    Placeholder implementation for tracing/compilation passes
    '''
    return


direct_register_custom_op(
    op_name="simamba_mixer",
    op_func=simamba_mixer,
    mutates_args=["output"],
    fake_impl=simamba_mixer_fake,
)
