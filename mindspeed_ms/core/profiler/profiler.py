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
"""pynative profile module"""
import os

from mindspore.profiler import DynamicProfilerMonitor, Profiler, ProfilerLevel
from mindspore.train import RunContext
from mindspore.communication import get_rank
from mindspeed_ms.tools import logger

from mindspeed_ms.training.global_vars import get_args


def get_real_rank():
    """get rank id"""
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


class ProfilerCallbackDict(dict):
    """Profiler Callback Dict, whether cur_step_num in self"""
    @property
    def cur_step_num(self):
        res = -1
        if "cur_step_num" in self:
            res = self["cur_step_num"]
        return res


class PynativeProfiler:
    r"""
    Pynative profiling class
    """

    def __init__(self):
        args = get_args()
        self.is_dynamic = False
        if args.profile:
            if args.profile_save_path:
                logger.warning(f"profile_save_path is not specified, using './profile' instead.")
            logger.info(f"profile will be saving to {args.profile_save_path}")
            if args.profile_dynamic_profiler_config_path:
                self.dynamic_profiler = DynamicProfilerMonitor(
                    cfg_path=args.profile_dynamic_profiler_config_path,
                    output_path=args.profile_save_path)
                self.is_dynamic = True
            else:
                profiler_level = None
                if args.profile_level == "level0":
                    profiler_level = ProfilerLevel.Level0
                elif args.profile_level == "level1":
                    profiler_level = ProfilerLevel.Level1
                elif args.profile_level == "level2":
                    profiler_level = ProfilerLevel.Level2
                logger.debug(f"profiler level {profiler_level}")

                profile_framework = None
                if args.profile_framework in ['all', 'time']:
                    profile_framework = args.profile_framework
                logger.debug(f"profile_framework {profile_framework}")

                # 按照rank_id设置性能数据落盘路径
                rank_id = get_real_rank()
                output_path = os.path.join(args.profile_save_path, f"rank_{rank_id}")

                self.profiler = Profiler(start_profile=False,
                                         output_path=output_path,
                                         profiler_level=profiler_level,
                                         with_stack=args.profile_with_stack,
                                         profile_memory=args.profile_memory,
                                         profile_framework=profile_framework,
                                         profile_communication=args.profile_communication,
                                         parallel_strategy=args.profile_parallel_strategy,
                                         aicore_metrics=args.profile_aicore_metrics,
                                         l2_cache=args.profile_l2_cache,
                                         hbm_ddr=args.profile_hbm_ddr,
                                         pcie=args.profile_pcie,
                                         data_process=args.profile_data_process,
                                         data_simplification=args.profile_data_simplification,
                                         op_time=args.profile_op_time)

    def step_begin(self, current_step):
        '''
        profiler step begin function
        Args:
            current_step (int): which step in training loop
        '''
        args = get_args()
        if not args.profile:
            return
        if self.is_dynamic:
            logger.info(f"start profiling in step {current_step}")
            cb_params = ProfilerCallbackDict({"cur_step_num": current_step})
            run_context = RunContext(cb_params)
            self.dynamic_profiler.step_begin(run_context)
        else:
            if current_step == args.profile_step_start:
                logger.info(f"start profiling in step {current_step}")
                self.profiler.start()

    def step_end(self, current_step):
        '''
        profiler step end function
        Args:
            current_step (int): which step in training loop
        '''
        args = get_args()
        if not args.profile:
            return
        if self.is_dynamic:
            cb_params = ProfilerCallbackDict({"cur_step_num": current_step})
            run_context = RunContext(cb_params)
            logger.info(f"end profiling in step {current_step}")
            self.dynamic_profiler.step_end(run_context)
        else:
            if current_step == args.profile_step_end:
                logger.info(f"stop profiling in step {current_step}")
                if args.profile_offline_analyse:
                    self.profiler.stop()
                else:
                    logger.info(f"analyzing profile")
                    self.profiler.analyse()
