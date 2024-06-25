# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from functools import wraps
from megatron.core import parallel_state
from megatron.training import get_args
from mindspeed.core.pipeline_parallel import flexible_schedules


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        if arguments.optimize_send_recv_comm and arguments.num_layers_per_virtual_pipeline_stage is None:
            return flexible_schedules.forward_backward_pipelining_without_interleaving
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 \
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None \
            and arguments.use_nanopipe:
            return flexible_schedules.forward_backward_pipelining_with_interleaving_nano_pipe
        return get_forward_backward_func(*args, **kwargs)
    return wrapper
