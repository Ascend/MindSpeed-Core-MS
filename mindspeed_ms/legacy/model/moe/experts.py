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
"""define experts"""

from mindspore import nn, ops

from mindspeed_ms.core.config import TransformerConfig
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.legacy.model.mlp import ParallelMLP


class GroupedMLP(Module):
    """
    Implement of Grouped MLP
    """

    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(GroupedMLP, self).__init__()

        self.num_local_experts = num_local_experts
        self.config = config

        raise NotImplementedError("GroupedMLP function not implemented.")


class SequentialMLP(Module):
    """
    Define SequentialMLP module.

    Args:
        num_local_experts (int): The number of local experts.
        config (TransformerConfig): Configuration object for the transformer model.
        submodules (MLPSubmodules): Type of the linear fully connected layer. Reserved parameter, currently not in use.

    Inputs:
        - **permuted_local_hidden_states** (Tensor) - The permuted input hidden states of the local experts.
        - **token_per_expert** (Tensor) - The number of tokens per expert.

    Outputs:
        Tuple of 2 Tensor.

        - **output_local** (Tensor) - The output of the local experts.
        - **output_bias_local** (Tensor) - The bias of the local experts. Currently not in use.
          The return value is ``None``.

    Raises:
        NotImplementedError: If `submodules` is not None.

    Examples:
        .. note::
            Before running the following examples, you need to configure the environment variables.

            For Ascend devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore.communication import init
        >>> from mindspeed_ms.core.config import TransformerConfig, ModelParallelConfig, TrainingConfig
        >>> from mindspeed_ms.legacy.model.moe.experts import SequentialMLP
        >>> from mindspeed_ms.core.parallel_state import initialize_model_parallel
        >>> ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic='ON')
        >>> init()
        >>> initialize_model_parallel()
        >>> parallel_config = ModelParallelConfig()
        >>> training_config = TrainingConfig(parallel_config=parallel_config)
        >>> config = TransformerConfig(vocab_size=128,
        ...                            num_layers=1,
        ...                            num_attention_heads=8,
        ...                            num_query_groups=4,
        ...                            hidden_size=64,
        ...                            ffn_hidden_size=128,
        ...                            parallel_config=parallel_config,
        ...                            training_config=training_config)
        >>> expert = SequentialMLP(num_local_experts=2, config=config)
        >>> shape = (32, 64)
        >>> tokens_per_expert = [20, 12]
        >>> permuted_local_hidden_states = ms.Tensor(np.random.standard_normal(shape).astype(np.float32))
        >>> output_local, output_bias_local = expert(permuted_local_hidden_states, tokens_per_expert)
        >>> print(output_local.shape)
        (32, 64)
    """
    def __init__(self, num_local_experts: int, config: TransformerConfig, submodules=None):
        super(SequentialMLP, self).__init__()
        if submodules is not None:
            raise NotImplementedError("`submodules` is not supported for now.")
        self.config = config
        self.num_local_experts = num_local_experts
        self.local_experts = nn.SequentialCell()
        for _ in range(self.num_local_experts):
            expert = ParallelMLP(self.config, is_expert=True)
            self.local_experts.append(expert)
        self.cast = ops.Cast()

    def construct(self, permuted_local_hidden_states, tokens_per_expert):
        """forward process"""
        if not permuted_local_hidden_states.shape:
            return permuted_local_hidden_states
        output_local = ops.zeros_like(permuted_local_hidden_states)

        start_idx = 0
        end_idx = tokens_per_expert[0]
        # every expert rank expert_id starts from `0`
        for expert_id, expert in enumerate(self.local_experts):
            hidden_expert = permuted_local_hidden_states[start_idx:end_idx]
            if hidden_expert.shape[0] == 0:
                output = hidden_expert
            else:
                output, _ = expert(hidden_expert)
            output_local[start_idx:end_idx] = output
            if expert_id != len(self.local_experts) - 1:
                start_idx = end_idx
                end_idx += tokens_per_expert[expert_id + 1]
        return output_local, None
