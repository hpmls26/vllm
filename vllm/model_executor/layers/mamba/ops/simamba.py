# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_local_simamba_on_path() -> bool:
    """
    Walk ancestor directories looking for a local simamba repo laid out as
    <root>/simamba/mamba_ssm/...  and insert <root>/simamba onto sys path
 
    Returns True if a candidate was found and inserted, False otherwise.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "simamba" / "mamba_ssm"
        if candidate.is_dir():
            simamba_root = str(candidate.parent)
            if simamba_root not in sys.path:
                sys.path.insert(0, simamba_root)
            return True

    return False


try:
    from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (
        mamba3_siso_combined as simamba_triton_siso_combined,
    )
    from mamba_ssm.ops.triton.simamba.mamba3_siso_step import (
        mamba3_siso_step as simamba_triton_siso_step,
    )
    from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (
        SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
        SIMAMBA_SUPPORTED_BOUNDARY_MODES,
        simamba_siso_combined,
        simamba_siso_step,
    )
except ImportError:
    _ensure_local_simamba_on_path()
    from mamba_ssm.ops.triton.simamba.mamba3_siso_combined import (
        mamba3_siso_combined as simamba_triton_siso_combined,
    )
    from mamba_ssm.ops.triton.simamba.mamba3_siso_step import (
        mamba3_siso_step as simamba_triton_siso_step,
    )
    from mamba_ssm.ops.triton.simamba.simamba_siso_combined import (
        SIMAMBA_BOUNDARY_MODE_ZERO_PAD,
        SIMAMBA_SUPPORTED_BOUNDARY_MODES,
        simamba_siso_combined,
        simamba_siso_step,
    )


__all__ = [
    "SIMAMBA_BOUNDARY_MODE_ZERO_PAD",
    "SIMAMBA_SUPPORTED_BOUNDARY_MODES",
    "simamba_siso_combined",
    "simamba_siso_step",
    "simamba_triton_siso_combined",
    "simamba_triton_siso_step",
]
