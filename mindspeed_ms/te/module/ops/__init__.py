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

"""Ops init."""

from mindspeed_ms.te.module.ops.default_ops import DefaultOps
from mindspeed_ms.training import get_args

OPS_MAP = {
    "default": DefaultOps
}


def get_ops():
    args = get_args()
    if not hasattr(args, 'ascend_turbo_type'):
        args.ascend_turbo_type = "default"
    return OPS_MAP[args.ascend_turbo_type]


class DummyHandle:

    def wait(self, *args, **kwargs):
        pass
