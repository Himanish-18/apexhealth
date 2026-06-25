"""
Healthcare Knowledge Navigator — Device Manager.

Automatic compute device detection with priority: CUDA → MPS → CPU.
Provides a unified interface for selecting and logging the active
hardware accelerator for embedding generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Information about the selected compute device.

    Attributes:
        device: The torch device object.
        device_type: Device category string ("cuda", "mps", or "cpu").
        device_name: Human-readable device name (e.g., "NVIDIA A100").
        gpu_memory_mb: Available GPU memory in MB, or None for CPU.
    """

    device: torch.device
    device_type: str
    device_name: str
    gpu_memory_mb: int | None


def detect_device(preferred: str | None = None) -> DeviceInfo:
    """Detect the best available compute device.

    Selection priority (unless overridden by ``preferred``):
        1. CUDA (NVIDIA GPU)
        2. MPS  (Apple Silicon GPU)
        3. CPU  (fallback)

    Args:
        preferred: Force a specific device type ("cuda", "mps", or "cpu").
                   If the requested device is unavailable, raises ValueError.

    Returns:
        DeviceInfo with the selected device details.

    Raises:
        ValueError: If the preferred device is not available or not recognized.
    """
    if preferred is not None:
        preferred = preferred.lower().strip()
        return _select_preferred(preferred)

    return _auto_detect()


def _auto_detect() -> DeviceInfo:
    """Auto-detect the best available device using CUDA → MPS → CPU priority."""
    if torch.cuda.is_available():
        return _make_cuda_info()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return _make_mps_info()

    return _make_cpu_info()


def _select_preferred(preferred: str) -> DeviceInfo:
    """Select a specific device, validating availability.

    Args:
        preferred: The requested device type.

    Returns:
        DeviceInfo for the requested device.

    Raises:
        ValueError: If the device is unavailable or unrecognized.
    """
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but not available. "
                "Ensure NVIDIA drivers and CUDA toolkit are installed."
            )
        return _make_cuda_info()

    if preferred == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError(
                "MPS device requested but not available. "
                "Requires macOS 12.3+ with Apple Silicon."
            )
        return _make_mps_info()

    if preferred == "cpu":
        return _make_cpu_info()

    raise ValueError(
        f"Unrecognized device type: '{preferred}'. "
        f"Supported values: 'cuda', 'mps', 'cpu'."
    )


def _make_cuda_info() -> DeviceInfo:
    """Build DeviceInfo for the default CUDA device."""
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    gpu_memory_mb = int(
        torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
    )

    info = DeviceInfo(
        device=device,
        device_type="cuda",
        device_name=device_name,
        gpu_memory_mb=gpu_memory_mb,
    )
    logger.info(
        "Selected device: CUDA — %s (%d MB VRAM), CUDA %s",
        device_name,
        gpu_memory_mb,
        torch.version.cuda or "N/A",
    )
    return info


def _make_mps_info() -> DeviceInfo:
    """Build DeviceInfo for the Apple MPS device."""
    device = torch.device("mps")

    info = DeviceInfo(
        device=device,
        device_type="mps",
        device_name="Apple Silicon GPU (MPS)",
        gpu_memory_mb=None,  # MPS does not expose memory info
    )
    logger.info("Selected device: MPS — Apple Silicon GPU")
    return info


def _make_cpu_info() -> DeviceInfo:
    """Build DeviceInfo for the CPU fallback."""
    info = DeviceInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        device_name="CPU",
        gpu_memory_mb=None,
    )
    logger.info("Selected device: CPU (no GPU acceleration)")
    return info
