"""Runtime helpers: device selection and backend availability."""

from __future__ import annotations

from .device import resolve_device, torch_device, torch_dtype

__all__ = ["resolve_device", "torch_device", "torch_dtype"]

