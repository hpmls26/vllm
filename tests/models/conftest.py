import pytest
from types import SimpleNamespace
import torch
from unittest.mock import patch, MagicMock

from vllm.model_executor.layers.mamba.simamba_mixer import SimambaMixer


def _make_fake_vllm_config():
    return SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={})
    )

@pytest.fixture
def mixer_fixture(): 
    fake_vllm_config = _make_fake_vllm_config()

    fake_tp_group = MagicMock()
    fake_tp_group.rank_in_group = 0
    fake_tp_group.world_size = 1

    with (
        # Patch the TP global so ColumnParallelLinear's get_tp_group() call
        # doesn't assert-fail on uninitialized process group
        patch("vllm.distributed.parallel_state._TP", fake_tp_group),

        # Patch get_current_vllm_config at every call site that gets hit
        # during SimambaMixer.__init__ — both in simamba_mixer.py itself
        # and inside ColumnParallelLinear / RowParallelLinear
        patch(
            "vllm.model_executor.layers.mamba.simamba_mixer.get_current_vllm_config",
            return_value=fake_vllm_config,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_current_vllm_config",
            return_value=fake_vllm_config,
        ),
        # Also patch it at the config module level in case anything calls
        # the canonical import path directly
        patch(
            "vllm.config.get_current_vllm_config",
            return_value=fake_vllm_config,
        ),

        # Patch TP rank/size in simamba_mixer's own namespace
        patch(
            "vllm.model_executor.layers.mamba.simamba_mixer"
            ".get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.mamba.simamba_mixer"
            ".get_tensor_model_parallel_world_size",
            return_value=1,
        ),

        # Suppress PluggableLayer.register so it doesn't touch distributed state
        patch(
            "vllm.model_executor.custom_op.PluggableLayer.register",
            side_effect=lambda name: (lambda cls: cls),
        ),
    ):
        mixer = SimambaMixer(
            hidden_size=512,
            ssm_state_size=128,
            expand=2,
            head_dim=64,
            n_groups=1,
            rope_fraction=0.5,
            use_midpoint_control=False,
            simamba_backend="reference",
            simpson_boundary_mode="zero_pad",
            is_outproj_norm=False,
            model_config=None,
            cache_config=None,
            prefix="test",
        )
        yield mixer


def test_finalize_accepts_shaped_z(mixer_fixture):
    mixer = mixer_fixture
    T = 4
    y = torch.zeros(T, mixer.local_d_inner)
    z_shaped = torch.zeros(T, mixer.local_num_heads, mixer.head_dim)

    out = mixer._finalize(y, z_shaped)
    assert out.shape == (T, mixer.hidden_size)


def test_project_returns_shaped_z(mixer_fixture):
    mixer = mixer_fixture
    T = 4
    hidden = torch.randn(T, mixer.hidden_size)

    p = mixer._project(hidden)

    assert p["z"].shape == (T, mixer.local_num_heads, mixer.head_dim), (
        f"Expected z shape ({T}, {mixer.local_num_heads}, {mixer.head_dim}), "
        f"got {p['z'].shape}"
    )
    assert p["x"].shape == (T, mixer.local_num_heads, mixer.head_dim), (
        f"Expected x shape ({T}, {mixer.local_num_heads}, {mixer.head_dim}), "
        f"got {p['x'].shape}"
    )
    assert p["b"].ndim == 2
    assert p["c"].ndim == 2