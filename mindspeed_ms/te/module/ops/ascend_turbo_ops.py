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

"""ascend turbo ops."""

from abc import ABC, abstractmethod


class AscendTurboConfig:
    save_allgather_input = True
    parallel_num = 2


class AscendTurboOps(ABC):
    """AscendTurboOps"""

    @staticmethod
    @abstractmethod
    def allgather_matmul(input_, weight, bias, fp8_meta=None, key=None):
        pass

    @staticmethod
    @abstractmethod
    def matmul_reduce_scatter(input_, weight, bias, fp8_meta=None, key=None):
        pass

    @staticmethod
    @abstractmethod
    def matmul_all_reduce(input_, weight, bias, fp8_meta=None, key=None):
        pass


ASCEND_TURBO_CONFIG = AscendTurboConfig()
