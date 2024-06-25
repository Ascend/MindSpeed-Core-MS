#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import functools
from megatron.core.distributed.param_and_grad_buffer import BufferType


def get(self, shape, start_index: int, buffer_type: BufferType):
    """
    Return a tensor with the input `shape` as a view into the 1-D data starting at
    `start_index`.
    """
    # end_index = start_index + shape.numel()
    numel = functools.reduce(lambda x, y: x * y, shape)
    end_index = start_index + numel
    if buffer_type.value == 1:
        buffer_tensor = self.param_data[start_index:end_index]
    elif buffer_type.value == 2:
        buffer_tensor = self.grad_data[start_index:end_index]
    else:
        raise Exception("Illegal buffer type provided to GradBuffer._get() function")
    buffer_tensor = buffer_tensor.view(shape)
    return buffer_tensor
