from typing import *  # noqa: F403
import torch


def _get_autocast_kwargs(device_type='cuda'):
    if torch.amp.is_autocast_available(device_type):
        device_autocast_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": False
        }
    else:
        device_autocast_kwargs = None
    cpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled('cpu'),
        "dtype": torch.get_autocast_dtype('cpu'),
        "cache_enabled": False
    }
    return device_autocast_kwargs, cpu_autocast_kwargs


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )
