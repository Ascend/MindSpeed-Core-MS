# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class MixtralParallelMLPBM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.ffn_hidden_size
        self.hidden_dim = config.hidden_size

        self.w1 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.w2 = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=False,
        )

        self.w3 = ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            is_expert=False,
        )

        self.act_fn = F.silu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)[0]) * self.w3(hidden_states)[0]
        current_hidden_states = self.w2(current_hidden_states)[0]
        return current_hidden_states
