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

"""FP8"""

from typing import Optional
from contextlib import contextmanager

from mindspeed_ms.te.fp8.state_manager import FP8GlobalStateManager


@contextmanager
def fp8_autocast(
        enabled: bool = True,
        fp8_config=None,
        calibrating: bool = False,
        fp8_group: Optional[str] = None,
        fp8_graph: bool = False,
):
    """auto cast for fp8"""
    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    FP8GlobalStateManager.fp8_autocast_enter(
        enabled=enabled,
        fp8_config=fp8_config,
        calibrating=calibrating,
        fp8_group=fp8_group,
        fp8_graph=fp8_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        FP8GlobalStateManager.fp8_autocast_exit(enabled, fp8_graph=fp8_graph)
