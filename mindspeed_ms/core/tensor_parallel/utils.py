# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

"""utility functions used throughout MindSpeed core"""

from typing import List

from mindspore import Tensor
import mindspore as ms
from mindspeed_ms.core.utils import divide


def split_tensor_along_last_dim(
        tensor: Tensor, num_partitions: int, contiguous_split_chunks: bool = False,) -> List[Tensor]:
    """ Split a tensor along its last dimension.

        Args:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.shape[last_dim], num_partitions)
    # Split.
    tensor_list = ms.mint.split(tensor, last_dim_size, dim=last_dim)
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
