# Copyright 2022 Huawei Technologies Co., Ltd
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

"""training init"""
from .arguments import parse_args, core_transformer_config_from_args
from .global_vars import get_args
from .global_vars import get_signal_handler
from .global_vars import get_tokenizer
from .global_vars import get_timers
from .global_vars import get_tensorboard_writer
from .global_vars import get_wandb_writer
from .global_vars import get_one_logger
from .model_parallel_config import ModelParallelConfig
from .training import TrainOneStepCell, pretrain, train, get_model, ParallelTrainingReducer, evaluate_and_print_results
from .loss_func import get_loss_func
from .yaml_arguments import core_transformer_config_from_yaml
from .utils import print_rank_0

__all__ = ["parse_args", "get_args", "core_transformer_config_from_args",
           "TrainOneStepCell", "train", "pretrain", 'get_model',
           "ModelParallelConfig", 'ParallelTrainingReducer', 'get_loss_func',
           "core_transformer_config_from_yaml", "evaluate_and_print_results"]
