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
"""MLP Module"""

from dataclasses import dataclass
from typing import Union

from mindspore import ops
from mindspeed_ms.core.transformer import ModuleSpec, build_module
from mindspeed_ms.legacy.model.activation import get_act_func
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.transformer import TransformerConfig


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(Module):
    r"""
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        is_expert (bool): This block is an expert block. Default: False.

    Inputs:
        - **hidden_states** (Tensor) - Tensor of shape :math:`(B, S, H)`.

    Outputs:
        - **output** (Tensor) - Output tensor of shape :math:`(B, S, H)`.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            is_expert: bool = False,
            input_size: int = None,
    ):
        super().__init__(config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size is not None else self.config.hidden_size

        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
        )

        self.activation_func = get_act_func(self.config.activation_func)

        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
        )

    def construct(self, hidden_states):
        """Construct function of mlp block."""
        # [B, S, H] -> [B, S, ffn_H] / [S, B, H] -> [S, B, ffn_H]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_activation_fusion:
            raise NotImplementedError("bias_activation_fusion not implemented for mlp.")

        if bias_parallel is not None:
            intermediate_parallel = intermediate_parallel + bias_parallel
        if self.config.gated_linear_unit:
            def glu(x):
                x = ops.chunk(x, 2, axis=-1)
                return self.activation_func(x[0]) * x[1]

            intermediate_parallel = glu(intermediate_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [B, S, ffn_H] -> [B, S, H] / [S, B, ffn_H] -> [S, B, H]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias
