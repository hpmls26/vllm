'''
Test class for verifying Simamba model integration in vLLM.

These tests validate:
    - Model registry alias resolution
    - Mamba state tensor contracts (shapes, dtypes, and copy functions)
'''

from types import SimpleNamespace

import torch

from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.simamba import SimambaForCausalLM


def _make_vllm_config():
    """
    Create a minimal synthetic vLLM configuration object.

    This avoids needing a full HuggingFace model config and instead provides 
    only the fields required by Simamba state utilities.
    """
    hf_config = SimpleNamespace(
        d_model=512,    
        n_layer=8,
        vocab_size=32000,
        tie_embeddings=True,
        ssm_cfg={
            "layer": "Simamba",
            "d_state": 128,          # State dimension of SSM
            "expand": 2,             # Expansion factor in hidden projections
            "headdim": 64,           # Attention/SSM head dimension
            "ngroups": 1,            # Number of grouped states
            "rope_fraction": 0.5,    # Fraction of rotary positional encoding
            "chunk_size": 64,
            "simamba_backend": "triton",
        },
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=hf_config,
            architecture="SimambaForCausalLM",
            model_impl="auto",
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",       # KV/cache dtype selection
            mamba_ssm_cache_dtype="auto",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


def test_simamba_registry_aliases_resolve():
    '''
    Ensure model registry aliases resolve correctly.
    Both "SimambaForCausalLM" and "MambaLMHeadModel" should resolve 
    to the same underlying implementation class.
    '''
    simamba_cls = ModelRegistry._try_load_model_cls("SimambaForCausalLM")
    mamba_lm_cls = ModelRegistry._try_load_model_cls("MambaLMHeadModel")

    assert simamba_cls is SimambaForCausalLM
    assert mamba_lm_cls is SimambaForCausalLM


def test_simamba_state_contract():
    """
    Validate that Simamba exposes a consistent internal state contract. 
    Specifically checks:
        - Number of state tensors
        - Shapes of each tensor
        - dtypes of each tensor
        - Presence of valid copy functions
    """
    vllm_config = _make_vllm_config()

    shapes = SimambaForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    dtypes = SimambaForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
    copy_funcs = SimambaForCausalLM.get_mamba_state_copy_func()

    # Check all expected state components exist
    assert len(shapes) == 6
    assert len(dtypes) == 6
    assert len(copy_funcs) == 6
    assert all(callable(copy_func) for copy_func in copy_funcs)
    # Verify exact expected tensor shapes
    assert shapes == (
        (8, 32),
        (8, 64, 128),
        (8, 128),
        (8, 128),
        (8, 64),
        (8, 64),
    )
    assert dtypes == (
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
    )
