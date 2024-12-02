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
"""HA Train classes and functions."""

import os
from functools import wraps
from enum import Enum
from typing import Callable
from mindspore import log as logger
from mindspeed_ms.core.parallel_state import get_data_parallel_group
from mindspeed_ms.core.dist_checkpointing import get_last_checkpoint
from .tft_adapter import MindIOAdapter, _CtxParam

class TrainPhase(Enum):
    TRAIN_START = 1
    TRAIN_STEP_START = 2
    OPT_UPDATE_START = 3
    OPT_UPDATE_END = 4
    TRAIN_STEP_END = 5
    TRAIN_END = 6

class _HATrainController:
    """
        Used for Mindspore High availability manager
    """
    def __init__(self):
        """
            Used for HA Train init
        """
        self.handlers = _CtxParam()
        self.enable = False

    def is_enable(self):
        return self.enable

    def init(self, **kwargs):
        """
            Used for HA Train init
        """
        self.handlers.tft = MindIOAdapter(**kwargs)
        self.enable = True

    def get_ha_ckpt(self, ckpt_dir, ckpt_format):
        dp_ranks = self._get_dp_ranks()
        for rank in dp_ranks:
            dir_tmp = os.path.join(ckpt_dir, f"rank_{rank}")
            if os.path.exists(dir_tmp):
                return get_last_checkpoint(dir_tmp, ckpt_format)
        return None

    def run_check(self):
        """
            Check if TFT is supported
        """
        return True

    def handle_exception(self, func: Callable):
        """
            Handle ha exceptions.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except SystemExit as e:    # pylint: disable=W0703
                logger.warning(f"catch system exit exception, {e}")
                raise e
            except TypeError as e:     # pylint: disable=W0703
                logger.warning(f"catch type error exception, {e}")
                raise e
            except BaseException as e: # pylint: disable=W0703
                logger.warning(f"Base exception caught: {e}")
                self.handlers.tft.report_tft_status()
                raise e
            self.handle_exit()

        return wrapper

    def update_status(self, run_phase: TrainPhase, **kwargs):
        """
            Handle update status.
        """
        if not self.enable:
            return
        if run_phase == TrainPhase.OPT_UPDATE_START:
            self.handlers.tft.set_start_updating_opt_state(**kwargs)

        if run_phase == TrainPhase.OPT_UPDATE_END:
            self.handlers.tft.train_step_end(**kwargs)

        if run_phase == TrainPhase.TRAIN_START:
            dp_ranks = self._get_dp_ranks()
            kwargs["dp_ranks"] = dp_ranks
            self.handlers.tft.register_processor(**kwargs)

    def handle_exit(self):
        """
            Handle exit.
        """
        self.handlers.tft.train_end()

    def _get_dp_ranks(self):
        data_group = get_data_parallel_group().split('-')
        dp_ranks = [int(rank) for rank in data_group if rank.isdigit()]
        return dp_ranks


ha_controller = _HATrainController()
