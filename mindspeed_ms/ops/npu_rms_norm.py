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

from mindspore.ops import rms_norm


def npu_rms_norm(x, gamma, epsilon=1e-6):
    """
    Inputs:
        - **x** (Tensor) - The input tensor.
        - **gamma** (Tensor) - Trainable parameter gamma.
        - **epsilon** (float) - An optional (0, 1] value used to avoid devide zero.

    Outputs:
        - **y** (Tensor) - The output of rms norm.
        - **rstd** (Tensor) - The reversed std.

    """
    out = rms_norm(x, gamma, epsilon)
    y, rstd = out

    return y, rstd
