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
"""moe layer"""
import mindspore as ms

from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.parallel_state import (
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_tensor_model_parallel_world_size
)
from mindspeed_ms.legacy.model.module import Module

from .experts import GroupedMLP, SequentialMLP
from .router import TopKRouter
from .token_dispatcher import MoEAlltoAllTokenDispatcher
from .pipe_expert_module import PipeExpertLayer
from .utils import token_sort, token_unsort


class MoELayer(Module):
    """
    Expert layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        submodules: reserve arguments, not used now.
        layer_number: reserve arguments, not used now.

    Inputs:
        - **hidden_states** (Tensor) - The input hidden states of the local experts.

    Outputs:
        Tuple of 2 Tensor.

        - **output* (Tensor) - The output of the local experts
        - **mlp_bias** (Tensor) - Not used now.

    Raises:
        ValueError: if `ep_world_size` is less than or equal to 0.
        ValueError: if `num_experts % ep_world_size` is not equal to 0.
        ValueError: if the elements of `local_expert_indices` is larger than or equal to `num_experts`.
        ValueError: if `moe_config.moe_token_dispatcher_type` is not "alltoall"
        ValueError: if `self.training` is true and `get_tensor_model_parallel_world_size()` is larger than 1,
            and `self.sp` is not true
    """
    # pylint: disable=C0103
    def __init__(self, config: TransformerConfig, submodules=None, layer_number: int = None):
        super(MoELayer, self).__init__()
        self.submodules = submodules
        self.layer_number = layer_number

        ep_world_size = get_expert_model_parallel_world_size()
        num_experts = config.num_moe_experts
        rank_id = get_expert_model_parallel_rank()

        self.tp = config.tensor_model_parallel_size
        self.sp = config.sequence_parallel

        if ep_world_size <= 0:
            raise ValueError(f"Expect expert parallel size > 0, but got {ep_world_size}")
        if num_experts % ep_world_size != 0:
            raise ValueError(f"Expect num_experts % ep_world_size == 0, but got {num_experts} and {ep_world_size}")

        num_local_experts = num_experts // ep_world_size
        local_expert_indices = [rank_id * num_local_experts + i for i in range(num_local_experts)]

        for x in local_expert_indices:
            if x >= num_experts:
                raise ValueError(f"expect all local expert indices < expert num, but got {local_expert_indices}")

        self.router = TopKRouter(config=config)
        self.router_topk = config.moe_router_topk
        self.use_pipe_expert_layer = config.use_pipe_expert_layer

        if config.moe_grouped_gemm:
            self.experts = GroupedMLP(num_local_experts, config)
        else:
            self.experts = SequentialMLP(num_local_experts, config)

        if config.moe_token_dispatcher_type == "alltoall":
            if self.use_pipe_expert_layer:
                self.pipe_expert_layer = PipeExpertLayer(
                    num_local_experts=num_local_experts,
                    local_expert_indices=local_expert_indices,
                    config=config
                )
            else:
                self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                    num_local_experts=num_local_experts,
                    local_expert_indices=local_expert_indices,
                    config=config
                    )
        else:
            raise ValueError(f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}")

    def construct(self, hidden_states: ms.Tensor):
        """moe layer forward"""
        if self.training and get_tensor_model_parallel_world_size() > 1 and not self.sp:
            raise ValueError(
                "During training, if tensor parallelism > 1 and not use sequence parallelism, "
                "would result in low performance in MoE."
            )
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.use_pipe_expert_layer:
            self.pipe_expert_layer.probs, self.pipe_expert_layer.indices = self.router(hidden_states)
            if self.pipe_expert_layer.probs.ndim != 2:
                raise ValueError(f"expect 'probs' is 2d tensor, \
                                 but got shape {self.pipe_expert_layer.probs.shape}.")
            if self.pipe_expert_layer.indices.ndim != 2:
                raise ValueError(f"expect 'indices' is 2d tensor, \
                                 but got shape {self.pipe_expert_layer.indices.shape}.")
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            sorted_local_input_tokens, local_input_sorted_map = token_sort(
                hidden_states,
                self.pipe_expert_layer.indices,
                topk=self.router_topk
            )
            total_input = []
            for expert_id in range(self.pipe_expert_layer.num_local_experts):
                total_input.append(self.experts.local_experts[expert_id].mapping.weight)
                total_input.append(self.experts.local_experts[expert_id].projection.weight)
            total_input.append(sorted_local_input_tokens)
            output_without_token_sorted = self.pipe_expert_layer(*total_input)

            output = token_unsort(
                sorted_tokens=output_without_token_sorted,
                sorted_indices=local_input_sorted_map,
                probs=self.pipe_expert_layer.probs,
                topk=self.router_topk
            )
        else:
            scores, indices = self.router(hidden_states)

            dispatched_input, tokens_per_expert = \
                self.token_dispatcher.token_permutation(hidden_states, scores, indices)
            expert_output, _ = self.experts(dispatched_input, tokens_per_expert)

            output, _ = self.token_dispatcher.token_unpermutation(expert_output, bias=None)
        output = output.reshape(hidden_shape)
        return output, None
