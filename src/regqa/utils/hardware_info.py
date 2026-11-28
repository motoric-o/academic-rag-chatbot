"""Hardware, platform, CUDA, and library version metadata collector.

Owner: Member C (Francisco Valentino; spec §0.4, §13 & docs/handoffs/a_to_c.md).
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


def _get_pkg_version(pkg_name: str) -> str | None:
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_hardware_info(
    device: str = "cpu",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect platform, CPU, GPU, and library version details for auditable runs."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "device": device,
        "gpu_name": None,
        "gpu_vram_total_mb": None,
        "cuda_version": None,
        "torch_version": _get_pkg_version("torch"),
        "transformers_version": _get_pkg_version("transformers"),
        "accelerate_version": _get_pkg_version("accelerate"),
        "bitsandbytes_version": _get_pkg_version("bitsandbytes"),
    }

    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_total_mb"] = round(props.total_memory / (1024 * 1024), 2)
    except Exception:
        pass

    if extra:
        info.update(extra)

    return info
