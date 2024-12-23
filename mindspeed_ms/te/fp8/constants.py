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
# ============================================================================

"""FP8 constants"""

from enum import Enum

import mindspore

# FP8 Dtype
if not hasattr(mindspore, 'float8_e4m3fn') or not hasattr(mindspore, 'float8_e5m2'):
    mindspore.float8_e4m3fn = mindspore.bfloat16
    mindspore.float8_e5m2 = mindspore.bfloat16


class FP8Format:
    def __init__(self, range_max: float, ebits: int, mbits: int, dtype: mindspore.dtype):
        self.max = range_max
        self.ebits = ebits
        self.mbits = mbits
        self.dtype = dtype


class Format(Enum):
    E4M3 = FP8Format(448, 4, 3, mindspore.float8_e4m3fn)
    E5M2 = FP8Format(57344, 5, 2, mindspore.float8_e5m2)


def amax_compute_max(amax, amax_history):
    amax.copy_(mindspore.ops.amax(amax_history, axis=0))


AMAX_COMPUTE_MAP = {
    'max': amax_compute_max
}
