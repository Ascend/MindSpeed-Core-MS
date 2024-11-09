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
from .model_parallel_config import ModelParallelConfig
from .training import TrainOneStepCell, pretrain, train, get_model, ParallelTrainingReducer
from .loss_func import get_loss_func
from .yaml_arguments import core_transformer_config_from_yaml

__all__ = ["parse_args", "get_args", "core_transformer_config_from_args",
           "TrainOneStepCell", "train", "pretrain", 'get_model',
           "ModelParallelConfig", 'ParallelTrainingReducer', 'get_loss_func',
           "core_transformer_config_from_yaml"]
