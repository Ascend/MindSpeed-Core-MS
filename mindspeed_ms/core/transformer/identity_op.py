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
"""Identity Op"""

from mindspore import nn


class IdentityOp(nn.Cell):
    """
    This is a placeholder for IdentityOp(x) -> x
    """

    # pylint: disable=W0613
    def __init__(self, *args, **kwargs):
        super().__init__()

    # pylint: disable=W0613
    def construct(self, x, *args, **kwargs):
        return x


class IdentityFuncOp(IdentityOp):
    """
    This is a placeholder for IdentityFuncOp(...)(x) -> IdentityOp(x) -> x.
    Such a func is handy for ops like `bias_dropout_fusion` which themselves
    return a function at runtime based on passed arguments
    """

    # pylint: disable=W0613
    def __init__(self, *args, **kwargs):
        super().__init__()

    # pylint: disable=W0613,W0221
    def construct(self, *args, **kwargs):
        return super().construct
