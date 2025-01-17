"""inner ops"""
import mindspore
from mindspore import ops
from ..configs import use_pyboost

def cast(input, dtype):
    return ops.cast(input, dtype)

def assign(input, other):
    return ops.assign(input, other)

def pad(input, pad, mode='constant', value=0.0):
    if use_pyboost():
        return mindspore.mint.nn.functional.pad(input, pad, mode, value)
    if mode == 'reflect':
        return ops.pad(input, pad, mode)
    return ops.pad(input, pad, mode, value)

def get_item(t):
    if isinstance(t, mindspore.Tensor):
        if t.numel() == 1:
            return t.item()
        else:
            raise ValueError("Cannot get item from tensor whose size is larger than 1")
    return t

__all__ = ['cast', 'assign']
