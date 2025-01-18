import mindspore
from typing import Sequence
from mindspore import Tensor, ops
from mindspore.common._stub_tensor import StubTensor
from mindspore._c_expression import Tensor as Tensor_
from .configs import use_pyboost

from ._utils import _rebuild_tensor_v2
from ._C.size import Size
from .ops import (transpose, mean, repeat_interleave, unsqueeze, pow, 
                  split, norm, reshape)
from .ops._inner import get_item

MS_PT_DTYPE_MAP = {
    'Float32': 'torch.cuda.FloatTensor',
    'BFloat16': 'torch.cuda.BFloat16Tensor',
    'Float16': 'torch.cuda.HalfTensor',
}

def npu(self):
    return Tensor(self.move_to("Ascend"))
Tensor.npu = npu
StubTensor.npu = npu

def data_ptr(self):
    return self._data_ptr()

Tensor.data_ptr = data_ptr
StubTensor.data_ptr = data_ptr

def type_(self, dtype=None):
    if dtype is None:
        return MS_PT_DTYPE_MAP[str(self.dtype)]
    return self.to(dtype)

Tensor.type = type_
StubTensor.type = type_

import weakref
def retain_grad(self):
    weak_self = weakref.ref(self)

    def _tensor_hook(grad):
        cur_self = weak_self()
        if cur_self is not None:
            cur_self.grad = grad
        return grad

    self.handle = self.register_hook(_tensor_hook)

Tensor.retain_grad = retain_grad
StubTensor.retain_grad = retain_grad

@property
def shape(self):
    if isinstance(self, StubTensor):
        if self.stub is not None:
            stub_shape = self.stub.get_shape()
        else:
            stub_shape = self.tensor.shape
        return Size(stub_shape)
    return Size(self._shape)

Tensor.shape = shape
StubTensor.shape = shape

def to_dense(self):
    return self

Tensor.to_dense = to_dense
StubTensor.to_dense = to_dense

Tensor._base = None
StubTensor._base = None

@property
def data(self):
    return self

@data.setter
def data(self, new_value):
    if isinstance(self, StubTensor) and isinstance(new_value, StubTensor):
        self.stub = new_value.stub
    else:
        self.assign_value(new_value)

Tensor.data = data
StubTensor.data = data

def numel(self):
    return ops.size(self)

Tensor.numel = numel
setattr(StubTensor, 'numel', numel)
Tensor.nelement = numel
StubTensor.nelement = numel

StubTensor.__hash__ = Tensor.__hash__

def _repeat(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], Sequence):
        sizes = sizes[0]
    sizes = tuple([get_item(t) for t in sizes])
    return ops.tile(self, tuple(sizes))

Tensor.repeat = _repeat
StubTensor.repeat = _repeat

def move_to_cuda(self, non_blocking=False):
    return Tensor(self.move_to("Ascend", not non_blocking))

def move_to_cpu(self, non_blocking=False):
    return Tensor(self.move_to("CPU", not non_blocking))

Tensor.cuda = move_to_cuda
StubTensor.cuda = move_to_cuda
Tensor.cpu = move_to_cpu
StubTensor.cpu = move_to_cpu


def size(self, dim=None):
    if dim is None:
        return self.shape
    assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
    return self.shape[dim]

Tensor.size = size
StubTensor.size = size

def dim(self):
    return self.ndim

Tensor.dim = dim
StubTensor.dim = dim

def clone(self):
    return self.copy()

Tensor.clone = clone
StubTensor.clone = clone

def log_softmax(self, dim=-1, dtype=None):
    if use_pyboost():
        return mindspore.mint.nn.functional.log_softmax(self, dim=dim, dtype=dtype)
    out = ops.log_softmax(self, dim)
    if dtype is not None:
        out = out.to(dtype)
    return out

Tensor.log_softmax = log_softmax
StubTensor.log_softmax = log_softmax

has_narrow = hasattr(mindspore.mint, 'narrow')
def narrow(self, dim, start, length):
    if not isinstance(start, int):
        start = start.item()
    if not isinstance(length, int):
        length = length.item()
    if use_pyboost() and has_narrow:
        return mindspore.mint.narrow(self, dim, start, length)
    return ops.narrow(self, dim, start, length)

Tensor.narrow = narrow
StubTensor.narrow = narrow

def view(self, *shape):
    result = []
    if type(shape) is tuple:
        for items in shape:
            if not isinstance(items, int):
                for item in items:
                    if not isinstance(item, int):
                        result.append(item.item())
                    else:
                        result.append(item)
            else:
                result.append(items)
    return ops.reshape(self, result)

Tensor.view = view
StubTensor.view = view


def __or__(self, other):
    if isinstance(other, (int, bool, float, Tensor)):
        return ops.bitwise_or(self.to(mindspore.int32), other.to(mindspore.int32)).bool()
    raise TypeError("Unsupported operand type(s) for |: 'Tensor' and '{}'".format(type(other)))

Tensor.__or__ = __or__
StubTensor.__or__ = __or__

Tensor.device = 'npu'
StubTensor.device = 'npu'

def mock_div_(self, value, *, rounding_mode=None):
    out = self.div(value, rounding_mode=rounding_mode)
    self.assign_value(out)

if not hasattr(Tensor, 'div_'):
    Tensor.div_ = mock_div_
    StubTensor.div_ = mock_div_

def __reduce_ex__(self, protocol):
    if isinstance(self, StubTensor):
        data = Tensor_(self.stub_sync())
    else:
        data = Tensor_(self)
    storage_offset = 0
    size = data._shape
    stride = data.stride()
    requires_grad = False
    args = (data, storage_offset, size, stride, requires_grad, None, None)
    return (
        _rebuild_from_type_v2, (_rebuild_tensor_v2, type(self), args, None))


Tensor.__reduce_ex__ = __reduce_ex__
StubTensor.__reduce_ex__ = __reduce_ex__

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret

def detach(self):
    return self

Tensor.detach = detach
StubTensor.detach = detach


Tensor.transpose = transpose
StubTensor.transpose = transpose


Tensor.mean = mean
StubTensor.mean = mean

Tensor.is_cuda = True
StubTensor.is_cuda = True

Tensor.repeat_interleave = repeat_interleave
StubTensor.repeat_interleave = repeat_interleave

def mul_(self, other):
    self.assign_value(self.mul(other))

Tensor.mul_ = mul_
StubTensor.mul_ = mul_
# Tensor.__mul__ = mul_
# StubTensor.__mul__ = mul_


Tensor.is_sparse = False
StubTensor.is_sparse = False

def requires_grad_(self, requires_grad=True):
    self.requires_grad = requires_grad
    return self

Tensor.requires_grad_ = requires_grad_
StubTensor.requires_grad_ = requires_grad_

Tensor.unsqueeze = unsqueeze
StubTensor.unsqueeze = unsqueeze

def __pow__(self, exponent):
    return pow(self, exponent)

Tensor.__pow__ = __pow__
StubTensor.__pow__ = __pow__

def _float(self):
    if self.dtype == mindspore.float32:
        return self
    return self.to(mindspore.float32)

Tensor.float = _float
StubTensor.float = _float

def expand(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], Sequence):
        sizes = sizes[0]
    sizes = tuple([get_item(t) for t in sizes])
    return self.broadcast_to(sizes)

Tensor.expand = expand
StubTensor.expand = expand

def split_(self, split_size, dim=0):
    return split(self, split_size, dim)

Tensor.split = split_
StubTensor.split = split_

def norm_(self, p='fro', dim=None, keepdim=False, dtype=None):
    return norm(self, p=p, dim=dim, keepdim=keepdim, dtype=dtype)

Tensor.norm = norm_
StubTensor.norm = norm_

def __imul__(self, other):
    self.copy_(self.mul(other))
    return self

Tensor.__imul__ = __imul__
StubTensor.__imul__ = __imul__

def cpu(self):
    return self

Tensor.cpu = cpu
StubTensor.cpu = cpu

def _reshape(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], Sequence):
        sizes = sizes[0]
    sizes = tuple([get_item(t) for t in sizes])
    return reshape(self, sizes)

Tensor.reshape = _reshape
StubTensor.reshape = _reshape