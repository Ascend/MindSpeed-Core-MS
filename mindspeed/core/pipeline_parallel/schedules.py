# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
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
import os
from functools import wraps
from megatron.training.arguments import parse_args
from mindspeed.arguments import parse_args_wrapper
from mindspeed.core.pipeline_parallel import flexible_schedules


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        forward_backward_func = get_forward_backward_func()
        new_parse_args = parse_args_wrapper(parse_args)
        arguments = new_parse_args(None, False)
        if arguments.optimize_send_recv_comm:
            forward_backward_func = flexible_schedules.forward_backward_pipelining_without_interleaving
        return forward_backward_func
    return wrapper