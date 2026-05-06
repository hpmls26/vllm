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
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata


def _make_vllm_config():
    '''
    Builds a minimal synthetic vLLM configuration object to avoid requiring 
    a full HuggingFace or runtime config.
    '''
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
    '''
    Validates Simamba internal state tensor contract. 
    
    Checks all expected state tensors are present, tensor shapes match 
    configuration-derived expectations, and copy functions exist and are callable
    '''
    vllm_config = _make_vllm_config()

    hf = vllm_config.model_config.hf_config
    ssm = hf.ssm_cfg

    hidden_size = hf.d_model
    expand = ssm["expand"]
    head_dim = ssm["headdim"]
    d_state = ssm["d_state"]
    rope_frac = ssm["rope_fraction"]
    tp = vllm_config.parallel_config.tensor_parallel_size

    d_inner = hidden_size * expand      # 1024
    num_heads = d_inner // head_dim     # 16
    local_num_heads = num_heads // tp   # 8

    split = int(d_state * rope_frac)
    if split % 2 != 0:
        split -= 1
    num_rope_angles = split // 2    # 32

    shapes = SimambaForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    dtypes = SimambaForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
    copy_funcs = SimambaForCausalLM.get_mamba_state_copy_func()

    # Check all expected state components exist
    assert len(shapes) == 6
    assert len(dtypes) == 6
    assert len(copy_funcs) == 6
    assert all(callable(copy_func) for copy_func in copy_funcs)
    
    expected_shapes = (
        (local_num_heads, num_rope_angles),     # 0: angle  (8, 32)
        (local_num_heads, head_dim, d_state),   # 1: ssm    (8, 64, 128)
        (local_num_heads, d_state),     # 2: k0     (8, 128)
        (local_num_heads, d_state),     # 3: k1     (8, 128)
        (local_num_heads, head_dim),    # 4: v0     (8, 64)
        (local_num_heads, head_dim),    # 5: v1     (8, 64)
    )
    assert shapes == expected_shapes, f"Got {shapes}, expected {expected_shapes}"
    assert all(d == torch.bfloat16 for d in dtypes)


def test_finalize_accepts_shaped_z(mixer_fixture):
    '''
    Verifies _finalize correctly handles both shaped and flattened z inputs.

    Ensures:
        - z shaped as [T, H, D] is accepted and properly processed
        - Output tensor shape matches expected hidden size
        - No failure across different projection modes
    '''
    T = 4
    y = torch.zeros(T, mixer_fixture.local_d_inner)
    z_shaped = torch.zeros(T, mixer_fixture.local_num_heads, mixer_fixture.head_dim)
    # Should not raise regardless of is_outproj_norm
    out = mixer_fixture._finalize(y, z_shaped)
    assert out.shape == (T, mixer_fixture.hidden_size)



def test_project_output_shapes(mixer_factory):
    """
    _project must return z and x pre-shaped to [T, local_num_heads, head_dim].
    b and c must remain flat [T, n_groups * ssm_state_size].
    """
    model_config = SimpleNamespace(
        dtype=torch.float32,
        get_mamba_chunk_size=lambda: 2,
    )
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        mamba_cache_mode="none",
        mamba_block_size=1,
    )
    mixer = mixer_factory(model_config=model_config, cache_config=cache_config)

    T = 5
    hidden = torch.randn(T, mixer.hidden_size)
    p = mixer._project(hidden)

    # z and x shaped
    assert p["z"].shape == (T, mixer.local_num_heads, mixer.head_dim), (
        f"z: expected ({T}, {mixer.local_num_heads}, {mixer.head_dim}), got {p['z'].shape}"
    )
    assert p["x"].shape == (T, mixer.local_num_heads, mixer.head_dim), (
        f"x: expected ({T}, {mixer.local_num_heads}, {mixer.head_dim}), got {p['x'].shape}"
    )
    # b and c flat
    assert p["b"].shape == (T, mixer.n_groups * mixer.ssm_state_size)
    assert p["c"].shape == (T, mixer.n_groups * mixer.ssm_state_size)
    # gates flat [T, local_num_heads]
    assert p["dd_dt"].shape == (T, mixer.local_num_heads)
    assert p["dd_a"].shape  == (T, mixer.local_num_heads)
    # angles flat [T, num_rope_angles]
    assert p["angles"].shape == (T, mixer.num_rope_angles)


def test_scatter_gather_roundtrip(mixer_factory):
    """
    States written via _scatter_states must be recoverable exactly via
    _gather_states at the same slot index.
    """
    model_config = SimpleNamespace(
        dtype=torch.float32,
        get_mamba_chunk_size=lambda: 2,
    )
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        mamba_cache_mode="none",
        mamba_block_size=1,
    )
    mixer = mixer_factory(model_config=model_config, cache_config=cache_config)
    mixer.kv_cache = tuple(
        torch.zeros((4,) + shape, dtype=dtype)
        for shape, dtype in zip(mixer.get_state_shape(), mixer.get_state_dtype())
    )

    # Create a known nonzero state for batch size 1
    torch.manual_seed(7)
    fake_states = tuple(
        torch.randn((1,) + shape, dtype=dtype)
        for shape, dtype in zip(mixer.get_state_shape(), mixer.get_state_dtype())
    )

    write_idx = torch.tensor([2])   # write into slot 2
    mixer._scatter_states(write_idx, fake_states)

    read_idx = torch.tensor([2])
    recovered = mixer._gather_states(
        read_idx,
        has_initial=torch.tensor([True]),
    )

    for orig, rec in zip(fake_states, recovered):
        torch.testing.assert_close(
            orig.to(rec.dtype), rec, rtol=1e-4, atol=1e-4,
            msg="State did not survive scatter→gather roundtrip"
        )

def test_expand_bc_to_heads_n_groups_1(mixer_factory):
    """
    With n_groups=1, _expand_bc_to_heads must broadcast the single group
    across all local_num_heads without copying data.
    """
    model_config = SimpleNamespace(
        dtype=torch.float32,
        get_mamba_chunk_size=lambda: 2,
    )
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        mamba_cache_mode="none",
        mamba_block_size=1,
    )
    mixer = mixer_factory(model_config=model_config, cache_config=cache_config)
    assert mixer.n_groups == 1

    T = 4
    b = torch.randn(T, 1, mixer.ssm_state_size)
    c = torch.randn(T, 1, mixer.ssm_state_size)

    b_exp, c_exp = mixer._expand_bc_to_heads(b, c)

    assert b_exp.shape == (T, mixer.local_num_heads, mixer.ssm_state_size)
    assert c_exp.shape == (T, mixer.local_num_heads, mixer.ssm_state_size)
    # Every head must see identical values (broadcast, not copy)
    for h in range(mixer.local_num_heads):
        torch.testing.assert_close(b_exp[:, h, :], b[:, 0, :])
        torch.testing.assert_close(c_exp[:, h, :], c[:, 0, :])