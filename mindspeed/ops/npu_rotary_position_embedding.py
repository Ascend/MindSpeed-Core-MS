# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
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

from mindspore.ops import rotary_position_embedding


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    """
    Inputs:
        - **x** (Tensor) - The input tensor.
        - **cos** (Tensor) - The input cos tensor.
        - **sin** (Tensor) - The input sin tensor.
        - **mode** (int) - Optional mode value: 0 rotate half, 1 rotate interleaved.

    Outputs:
        - **y** (Tensor) - The output tensor.

    """
    return rotary_position_embedding(x, cos, sin, mode)