"""
Tests for device detection (embeddings.device_manager).
Tests auto-detection, forced device selection, and error handling.
"""

from unittest.mock import patch

import pytest
import torch

from embeddings.device_manager import DeviceInfo, detect_device


def test_detect_device_returns_device_info():
    """Auto-detection should always return a valid DeviceInfo."""
    info = detect_device()

    assert isinstance(info, DeviceInfo)
    assert isinstance(info.device, torch.device)
    assert info.device_type in ("cuda", "mps", "cpu")
    assert isinstance(info.device_name, str)
    assert len(info.device_name) > 0


def test_cpu_fallback():
    """Forcing CPU should return a CPU DeviceInfo."""
    info = detect_device(preferred="cpu")

    assert info.device_type == "cpu"
    assert info.device_name == "CPU"
    assert info.gpu_memory_mb is None
    assert info.device == torch.device("cpu")


def test_invalid_device_raises():
    """Requesting an unrecognized device should raise ValueError."""
    with pytest.raises(ValueError, match="Unrecognized device type"):
        detect_device(preferred="tpu")


def test_cuda_unavailable_raises():
    """Requesting CUDA when unavailable should raise ValueError."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(ValueError, match="CUDA device requested"):
            detect_device(preferred="cuda")


def test_preferred_case_insensitive():
    """Device selection should be case-insensitive."""
    info = detect_device(preferred="CPU")
    assert info.device_type == "cpu"

    info = detect_detect = detect_device(preferred="  cpu  ")
    assert info.device_type == "cpu"


def test_auto_detect_falls_to_cpu_when_no_gpu():
    """When no GPU is available, auto-detect should fall back to CPU."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        # Also patch MPS if available
        with patch("embeddings.device_manager.torch.backends") as mock_backends:
            mock_backends.mps.is_available.return_value = False
            # hasattr check needs to still work
            info = detect_device()
            assert info.device_type == "cpu"
