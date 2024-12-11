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
"""moe pipe expert layer"""
from typing import List
import mindspore as ms # type: ignore
import mindspore.communication.comm_func as comm_func
from mindspore import ops, mint, nn
from mindspore.communication import get_group_size
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig
from mindspeed_ms.core.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_world_size
)
from mindspeed_ms.core.tensor_parallel.mappings import (
    gather_along_first_dim_expert_parallel
)
from .utils import get_input_tokens, silu_grad


class PipeExpertLayerUtil:
    """Pipe expert layer util"""
    first_a2a_event = []
    second_a2a_event = []
    computing_mm_event = []

    ep_world_size = None
    ep_group = None

    @classmethod
    def get_first_a2a_event(cls):
        return cls.first_a2a_event

    @classmethod
    def get_second_a2a_event(cls):
        return cls.second_a2a_event

    @classmethod
    def get_computing_mm_event(cls):
        return cls.computing_mm_event

    @classmethod
    def async_a2a(
            cls,
            alltoall_input,
            output_shape,
            output_splits,
            input_splits,
            async_op,
            is_first
    ):
        """async alltoall communication"""
        alltoall_input = alltoall_input.contiguous()
        if cls.ep_world_size == 1:
            output = ops.stop_gradient(alltoall_input)
        else:
            output, handle = comm_func.all_to_all_single_with_output_shape(
                output_shape=output_shape,
                tensor=alltoall_input,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=cls.ep_group,
                async_op=async_op
            )
            if is_first:
                cls.first_a2a_event.append(handle)
            else:
                cls.second_a2a_event.append(handle)
        return output


class PipeExpertLayer(nn.Cell):
    """
    Pipe Expert Layer.

    Args:
        num_local_experts (int): The number of the local experts in every npu.
        local_expert_indices (list): The indices of local experts in every npu.
        config (TransformerConfig): Configuration object for the transformer model.
        transpose_b (bool): Transpose or not for batchmatmul.

    Inputs:
        - **total_input** (List) - list of total input.

    Outputs:
        - **output_without_token_unsorted** (Tensor)- The output tensor of the second alltoall.

    Raises:
        NotImplementedError: If `config.clone_scatter_output_in_embedding` is True.
        RuntimeError: If `tokentype_ids` is not None and `tokentype_embeddings` is None.
            If `tokentype_ids` is None and `tokentype_embeddings` is not None.
    """
    def __init__(
            self,
            num_local_experts: int,
            local_expert_indices: List[int],
            config: TransformerConfig,
            transpose_b=True
    ):
        super(PipeExpertLayer, self).__init__()

        self.config = config
        self.hidden_shape = None
        self.hidden_size = self.config.hidden_size
        self.num_local_experts = num_local_experts
        self.use_pipe_expert_recompute = self.config.use_pipe_expert_recompute
        self.use_pipe_expert_swap = self.config.use_pipe_expert_swap
        if self.use_pipe_expert_recompute and self.use_pipe_expert_swap:
            raise ValueError(f"Do not support recompute and swap at the same time now !!!")
        if self.num_local_experts <= 0:
            raise ValueError(f"expect num_local_experts > 0, but got {self.num_local_experts}.")

        self.transpose_b = transpose_b
        self.moe_router_topk = self.config.moe_router_topk
        self.add_bias = self.config.add_bias_linear
        self.num_moe_experts = self.config.num_moe_experts
        self.ep_size = self.config.expert_model_parallel_size
        self.use_sp = self.config.sequence_parallel
        self.local_expert_indices = local_expert_indices
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError(f"expect len(self.local_expert_indices) == {self.num_local_experts}, "
                             f"but got {len(self.local_expert_indices)}")

        expert_ids_per_ep_rank = [i % self.num_local_experts for i in range(self.num_moe_experts)]
        self.expert_ids_per_ep_rank = ms.Tensor(expert_ids_per_ep_rank, dtype=ms.int32)

        self.probs = None
        self.hidden_act = self.config.activation_func
        self.matmul_transpose_b = ops.operations.BatchMatMul(transpose_b=self.transpose_b)
        self.matmul_grad_transpose_a = ops.operations.BatchMatMul(transpose_a=self.transpose_b)
        self.matmul_notranspose = ops.operations.BatchMatMul()
        self.swap_stream = ms.hal.Stream()

        self.ep_group = get_expert_model_parallel_group()
        self.rank_id = get_expert_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        # variables for 'preprocess'
        self.indices = None

        # for bprop
        self.columnparallerlinear_input_list = []
        self.columnparallerlinear_output0_list = []
        self.columnparallerlinear_output1_list = []
        if not self.use_pipe_expert_recompute:
            self.columnparallerlinear_output0_silu_list = []
            self.mul_output_list = []

    def preprocess(self):
        """preprocess for alltoall"""
        ep_size, num_moe_experts = self.ep_size, self.num_moe_experts
        indices = self.indices

        count_local_tokens_per_expert = ops.histc(indices, bins=num_moe_experts, min=0, max=num_moe_experts)

        # cal input splits
        local_input_splits = count_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
        self.input_dim0_local_expert = local_input_splits.sum(axis=0).to(ms.int32).tolist()
        self.local_input_splits_per_expert = \
            count_local_tokens_per_expert.reshape(ep_size, self.num_local_experts).T.tolist()
        local_input_splits = local_input_splits.sum(axis=1).to(ms.int32)
        self.local_input_splits = local_input_splits.asnumpy().tolist()
        dp_group_input_splits = gather_along_first_dim_expert_parallel(local_input_splits)

        self.dp_group_input_splits = dp_group_input_splits.reshape(-1, ep_size)
        # cal output splits
        count_dp_group_tokens_per_expert = gather_along_first_dim_expert_parallel(count_local_tokens_per_expert)
        count_dp_group_tokens_per_expert = count_dp_group_tokens_per_expert.reshape(ep_size, num_moe_experts)
        self.count_dp_group_tokens_per_local_expert = count_dp_group_tokens_per_expert[:, self.local_expert_indices]
        self.output_dim0_local_expert = \
            self.count_dp_group_tokens_per_local_expert.sum(axis=0).to(ms.int32).tolist()
        self.split_index_presum = [0]
        for i in range(self.ep_size):
            for j in range(self.num_local_experts):
                self.split_index_presum.append(
                    (self.split_index_presum[-1]+self.local_input_splits_per_expert[j][i])
                )


    def construct(self, *total_input):
        """pipe expert layer forward"""
        if self.use_pipe_expert_swap:
            return self.swap_construct(*total_input)
        sorted_local_input_tokens = total_input[-1]
        self.preprocess()

        tokens_per_expert_different_rank_before_first_a2a = []
        tokens_per_expert_different_rank_after_second_a2a = ms.Tensor(0)
        for i in range(self.num_local_experts):
            tokens = get_input_tokens(
                sorted_local_input_tokens=sorted_local_input_tokens,
                local_input_splits_per_expert=self.local_input_splits_per_expert,
                split_index_presum=self.split_index_presum,
                expert_index=i
            )
            if tokens != []:
                tokens_per_expert_different_rank_before_first_a2a.append(tokens)

        tokens_per_expert_different_rank_before_first_a2a = \
            mint.cat(tokens_per_expert_different_rank_before_first_a2a).reshape(-1, self.hidden_size)

        PipeExpertLayerUtil.ep_group = self.ep_group
        PipeExpertLayerUtil.ep_world_size = get_group_size(group=self.ep_group)

        start_index = 0
        for alltoall_id in range(self.num_local_experts):
            output_shape_cur_alltoall = (
                self.output_dim0_local_expert[alltoall_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[alltoall_id].asnumpy().tolist()
            input_splits_cur_alltoall = \
                self.local_input_splits_per_expert[alltoall_id]

            end_index = self.input_dim0_local_expert[alltoall_id] + start_index

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=tokens_per_expert_different_rank_before_first_a2a[start_index:end_index],
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=True
            )
            if output.ndim != 0:
                self.columnparallerlinear_input_list.append(output)
            start_index = end_index

        mm_input_start_index = 0
        mm_input_tokens_index = 0
        for expert_id in range(self.num_local_experts):
            PipeExpertLayerUtil.first_a2a_event[expert_id].wait()
            mm_input_end_index = self.output_dim0_local_expert[expert_id] + mm_input_start_index

            if mm_input_end_index == mm_input_start_index:
                r_output = ms.Tensor(0.0)
            else:
                c_output0, c_output1, c_output0_silu, m_output, r_output = self.construct_comp(
                    token_index=mm_input_tokens_index,
                    col_weight=total_input[expert_id * 2],
                    row_weight=total_input[expert_id * 2 + 1]
                )
                mm_input_tokens_index += 1

            # second alltoall
            output_shape_cur_alltoall = (
                self.input_dim0_local_expert[expert_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.local_input_splits_per_expert[expert_id]
            input_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[expert_id].tolist()

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=r_output,
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=False
            )
            if output.ndim != 0:
                if tokens_per_expert_different_rank_after_second_a2a.ndim == 0:
                    tokens_per_expert_different_rank_after_second_a2a = output
                else:
                    tokens_per_expert_different_rank_after_second_a2a = \
                        mint.cat((tokens_per_expert_different_rank_after_second_a2a, output))

            if (self.hidden_act == 'swiglu' or self.hidden_act == 'silu') and mm_input_end_index > mm_input_start_index:
                self.columnparallerlinear_output0_list.append(c_output0)
                self.columnparallerlinear_output1_list.append(c_output1)
                if not self.use_pipe_expert_recompute:
                    self.columnparallerlinear_output0_silu_list.append(c_output0_silu)
                    self.mul_output_list.append(m_output)
            mm_input_start_index = mm_input_end_index

        PipeExpertLayerUtil.second_a2a_event[-1].wait()

        self.clean(forward=True)

        output_without_token_unsort = self.get_output(
            max_index=tokens_per_expert_different_rank_after_second_a2a.shape[0],
            input_tensor=tokens_per_expert_different_rank_after_second_a2a
        )

        return output_without_token_unsort


    def swap_construct(self, *total_input):
        """pipe expert layer forward when using swap"""
        sorted_local_input_tokens = total_input[-1]
        self.preprocess()

        tokens_per_expert_different_rank_before_first_a2a = []
        tokens_per_expert_different_rank_after_second_a2a = ms.Tensor(0)
        for i in range(self.num_local_experts):
            tokens = get_input_tokens(
                sorted_local_input_tokens=sorted_local_input_tokens,
                local_input_splits_per_expert=self.local_input_splits_per_expert,
                split_index_presum=self.split_index_presum,
                expert_index=i
            )
            if tokens != []:
                tokens_per_expert_different_rank_before_first_a2a.append(tokens)

        tokens_per_expert_different_rank_before_first_a2a = \
            mint.cat(tokens_per_expert_different_rank_before_first_a2a).reshape(-1, self.hidden_size)

        PipeExpertLayerUtil.ep_group = self.ep_group
        PipeExpertLayerUtil.ep_world_size = get_group_size(group=self.ep_group)

        start_index = 0
        for alltoall_id in range(self.num_local_experts):
            output_shape_cur_alltoall = (
                self.output_dim0_local_expert[alltoall_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[alltoall_id].asnumpy().tolist()
            input_splits_cur_alltoall = \
                self.local_input_splits_per_expert[alltoall_id]

            end_index = self.input_dim0_local_expert[alltoall_id] + start_index

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=tokens_per_expert_different_rank_before_first_a2a[start_index:end_index],
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=True
            )
            if output.ndim != 0:
                self.columnparallerlinear_input_list.append(output)
            start_index = end_index

        mm_input_start_index = 0
        mm_input_tokens_index = 0
        for expert_id in range(self.num_local_experts):
            PipeExpertLayerUtil.first_a2a_event[expert_id].wait()
            mm_input_end_index = self.output_dim0_local_expert[expert_id] + mm_input_start_index

            if mm_input_end_index == mm_input_start_index:
                r_output = ms.Tensor(0.0)
            else:
                c_output0, c_output1, c_output0_silu, m_output, r_output = self.construct_comp(
                    token_index=mm_input_tokens_index,
                    col_weight=total_input[expert_id * 2],
                    row_weight=total_input[expert_id * 2 + 1]
                )
                mm_input_tokens_index += 1

            # second alltoall
            output_shape_cur_alltoall = (
                self.input_dim0_local_expert[expert_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.local_input_splits_per_expert[expert_id]
            input_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[expert_id].tolist()

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=r_output,
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=False
            )
            if output.ndim != 0:
                if tokens_per_expert_different_rank_after_second_a2a.ndim == 0:
                    tokens_per_expert_different_rank_after_second_a2a = output
                else:
                    tokens_per_expert_different_rank_after_second_a2a = \
                        mint.cat((tokens_per_expert_different_rank_after_second_a2a, output))

            if (self.hidden_act == 'swiglu' or self.hidden_act == 'silu') and mm_input_end_index > mm_input_start_index:
                current_stream = ms.hal.current_stream()
                self.swap_stream.wait_stream(current_stream)
                with ms.hal.StreamCtx(self.swap_stream):
                    self.columnparallerlinear_output0_list.append(c_output0.move_to("CPU"))
                    self.columnparallerlinear_output1_list.append(c_output1.move_to("CPU"))

                self.columnparallerlinear_output0_silu_list.append(c_output0_silu)
                self.mul_output_list.append(m_output)
            mm_input_start_index = mm_input_end_index

        PipeExpertLayerUtil.second_a2a_event[-1].wait()

        self.clean(forward=True)

        output_without_token_unsort = self.get_output(
            max_index=tokens_per_expert_different_rank_after_second_a2a.shape[0],
            input_tensor=tokens_per_expert_different_rank_after_second_a2a
        )

        return output_without_token_unsort


    def bprop(self, *total_input):
        """pipe expert layer backward"""
        if self.use_pipe_expert_swap:
            return self.swap_bprop(*total_input)
        d_output_without_token_unsort = total_input[-1]

        grads_per_expert_different_rank_after_second_a2a = []
        grads_per_expert_different_rank_after_mm = ms.Tensor(0)
        grads_per_expert_different_rank_before_first_a2a = ms.Tensor(0)
        weights_grad = []

        for i in range(self.num_local_experts):
            grads = get_input_tokens(
                sorted_local_input_tokens=d_output_without_token_unsort,
                local_input_splits_per_expert=self.local_input_splits_per_expert,
                split_index_presum=self.split_index_presum,
                expert_index=i
            )
            if grads != []:
                grads_per_expert_different_rank_after_second_a2a.append(grads)

        self.grads_per_expert_different_rank_after_second_a2a = \
            mint.cat(grads_per_expert_different_rank_after_second_a2a).reshape(-1, self.hidden_size)

        # second alltoall bprop
        start_index = 0
        for alltoall_id in range(self.num_local_experts):
            output_shape_cur_alltoall = (
                self.output_dim0_local_expert[alltoall_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[alltoall_id].asnumpy().tolist()
            input_splits_cur_alltoall = \
                self.local_input_splits_per_expert[alltoall_id]

            end_index = self.input_dim0_local_expert[alltoall_id] + start_index

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=self.grads_per_expert_different_rank_after_second_a2a[start_index:end_index],
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=True
            )
            if output.ndim != 0:
                if grads_per_expert_different_rank_after_mm.ndim == 0:
                    grads_per_expert_different_rank_after_mm = output
                else:
                    grads_per_expert_different_rank_after_mm = \
                        ops.cat((grads_per_expert_different_rank_after_mm, output))
            start_index = end_index

        mm_input_start_index = 0
        activation_index = 0
        for expert_id in range(self.num_local_experts):
            PipeExpertLayerUtil.first_a2a_event[expert_id].wait()
            mm_input_end_index = self.output_dim0_local_expert[expert_id] + mm_input_start_index

            if mm_input_end_index == mm_input_start_index:
                column_weight_grad = ops.zeros(total_input[expert_id * 2].shape, ms.float32)
                row_weight_grad = ops.zeros(total_input[expert_id * 2 + 1].shape, ms.float32)
                output_grad = ms.Tensor(0.0)
            else:
                row_weight_grad, column_weight_grad, output_grad = self.bprop_comp(
                    token_index=activation_index,
                    input_grad=grads_per_expert_different_rank_after_mm[mm_input_start_index:mm_input_end_index],
                    col_weight=total_input[expert_id * 2],
                    row_weight=total_input[expert_id * 2 + 1]
                )
                activation_index += 1
            weights_grad.append(column_weight_grad)
            weights_grad.append(row_weight_grad)
            mm_input_start_index = mm_input_end_index

            # first alltoall bprop
            output_shape_cur_alltoall = (
                self.input_dim0_local_expert[expert_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.local_input_splits_per_expert[expert_id]
            input_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[expert_id].asnumpy().tolist()

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=output_grad,
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=False
            )
            if output.ndim != 0:
                if grads_per_expert_different_rank_before_first_a2a.ndim == 0:
                    grads_per_expert_different_rank_before_first_a2a = output
                else:
                    grads_per_expert_different_rank_before_first_a2a = \
                        ops.cat((grads_per_expert_different_rank_before_first_a2a, output))

        PipeExpertLayerUtil.second_a2a_event[-1].wait()

        self.clean(forward=False)

        sorted_local_input_tokens_grad = self.get_output(
            max_index=grads_per_expert_different_rank_before_first_a2a.shape[0],
            input_tensor=grads_per_expert_different_rank_before_first_a2a
        )

        weights_grad.append(sorted_local_input_tokens_grad)
        return weights_grad


    def swap_bprop(self, *total_input):
        """pipe expert layer backward when using swap"""
        d_output_without_token_unsort = total_input[-1]

        grads_per_expert_different_rank_after_second_a2a = []
        grads_per_expert_different_rank_after_mm = ms.Tensor(0)
        grads_per_expert_different_rank_before_first_a2a = ms.Tensor(0)
        weights_grad = []

        for i in range(self.num_local_experts):
            grads = get_input_tokens(
                sorted_local_input_tokens=d_output_without_token_unsort,
                local_input_splits_per_expert=self.local_input_splits_per_expert,
                split_index_presum=self.split_index_presum,
                expert_index=i
            )
            if grads != []:
                grads_per_expert_different_rank_after_second_a2a.append(grads)

        self.grads_per_expert_different_rank_after_second_a2a = \
            mint.cat(grads_per_expert_different_rank_after_second_a2a).reshape(-1, self.hidden_size)

        # second alltoall bprop
        start_index = 0
        for alltoall_id in range(self.num_local_experts):
            output_shape_cur_alltoall = (
                self.output_dim0_local_expert[alltoall_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[alltoall_id].asnumpy().tolist()
            input_splits_cur_alltoall = \
                self.local_input_splits_per_expert[alltoall_id]

            end_index = self.input_dim0_local_expert[alltoall_id] + start_index

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=self.grads_per_expert_different_rank_after_second_a2a[start_index:end_index],
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=True
            )
            if output.ndim != 0:
                if grads_per_expert_different_rank_after_mm.ndim == 0:
                    grads_per_expert_different_rank_after_mm = output
                else:
                    grads_per_expert_different_rank_after_mm = \
                        ops.cat((grads_per_expert_different_rank_after_mm, output))
            start_index = end_index

        mm_input_start_index = 0
        activation_index = 0
        for expert_id in range(self.num_local_experts):
            PipeExpertLayerUtil.first_a2a_event[expert_id].wait()
            mm_input_end_index = self.output_dim0_local_expert[expert_id] + mm_input_start_index

            if mm_input_end_index == mm_input_start_index:
                column_weight_grad = ops.zeros(total_input[expert_id * 2].shape, ms.float32)
                row_weight_grad = ops.zeros(total_input[expert_id * 2 + 1].shape, ms.float32)
                output_grad = ms.Tensor(0.0)
            else:
                row_weight_grad, column_weight_grad, output_grad = self.swap_bprop_comp(
                    token_index=activation_index,
                    input_grad=grads_per_expert_different_rank_after_mm[mm_input_start_index:mm_input_end_index],
                    col_weight=total_input[expert_id * 2],
                    row_weight=total_input[expert_id * 2 + 1]
                )
                activation_index += 1
            weights_grad.append(column_weight_grad)
            weights_grad.append(row_weight_grad)
            mm_input_start_index = mm_input_end_index

            # first alltoall bprop
            output_shape_cur_alltoall = (
                self.input_dim0_local_expert[expert_id],
                self.hidden_size // self.tp_size
            )
            output_splits_cur_alltoall = \
                self.local_input_splits_per_expert[expert_id]
            input_splits_cur_alltoall = \
                self.count_dp_group_tokens_per_local_expert.T[expert_id].asnumpy().tolist()

            output = PipeExpertLayerUtil.async_a2a(
                alltoall_input=output_grad,
                output_shape=output_shape_cur_alltoall,
                output_splits=output_splits_cur_alltoall,
                input_splits=input_splits_cur_alltoall,
                async_op=True,
                is_first=False
            )
            if output.ndim != 0:
                if grads_per_expert_different_rank_before_first_a2a.ndim == 0:
                    grads_per_expert_different_rank_before_first_a2a = output
                else:
                    grads_per_expert_different_rank_before_first_a2a = \
                        ops.cat((grads_per_expert_different_rank_before_first_a2a, output))

        PipeExpertLayerUtil.second_a2a_event[-1].wait()

        self.clean(forward=False)

        sorted_local_input_tokens_grad = self.get_output(
            max_index=grads_per_expert_different_rank_before_first_a2a.shape[0],
            input_tensor=grads_per_expert_different_rank_before_first_a2a
        )

        weights_grad.append(sorted_local_input_tokens_grad)
        return weights_grad


    def get_output(self, max_index, input_tensor):
        """sort the output of the second alltoall"""
        row = 0
        index_list = [0] * max_index
        for expert_id in range(self.num_local_experts):
            for k in range(self.ep_size):
                index_start = self.split_index_presum[expert_id + k * self.num_local_experts]
                index_end = self.split_index_presum[expert_id + k * self.num_local_experts + 1]
                if index_start == index_end:
                    continue
                index_list[index_start:index_end] = \
                    list(range(row, row+self.local_input_splits_per_expert[expert_id][k]))
                row += self.local_input_splits_per_expert[expert_id][k]
        output = mint.index_select(input_tensor, 0, ms.Tensor(index_list))
        return output


    def clean(self, forward):
        """clean the list of events and activations"""
        PipeExpertLayerUtil.first_a2a_event.clear()
        PipeExpertLayerUtil.second_a2a_event.clear()
        PipeExpertLayerUtil.computing_mm_event.clear()
        if not forward:
            self.columnparallerlinear_input_list.clear()
            self.columnparallerlinear_output0_list.clear()
            self.columnparallerlinear_output1_list.clear()
            if not self.use_pipe_expert_recompute:
                self.columnparallerlinear_output0_silu_list.clear()
                self.mul_output_list.clear()


    def construct_comp(self, token_index, col_weight, row_weight):
        """expert computing in forward process"""
        # ColumnParallerLinear
        c_output = self.matmul_transpose_b(
            self.columnparallerlinear_input_list[token_index],
            col_weight
        )

        # activation function
        if self.hidden_act == 'swiglu' or self.hidden_act == 'silu':
            c_output0, c_output1 = mint.split(c_output, c_output.shape[-1]//2, dim=-1)
            c_output0_silu = c_output0 / (1 + mint.exp(-c_output0))
            m_output = c_output0_silu * c_output1
        else:
            raise ValueError(f"Only support the act_function 'SwiGlu' or 'silu' now, but got {self.hidden_act}.")

        # RowParallerLinear
        r_output = self.matmul_transpose_b(m_output, row_weight)

        computing_event = ms.hal.Event()
        computing_event.record()
        PipeExpertLayerUtil.computing_mm_event.append(computing_event)

        return c_output0, c_output1, c_output0_silu, m_output, r_output


    def bprop_comp(self, token_index, input_grad, col_weight, row_weight):
        """computing expert gradients in backward process"""
        # RowParallerLinear Grad
        m_output_grad = self.matmul_notranspose(input_grad, row_weight)
        if not self.use_pipe_expert_recompute:
            row_weight_grad = self.matmul_grad_transpose_a(
                input_grad,
                self.mul_output_list[token_index]
            )
        else:
            c_output0 = self.columnparallerlinear_output0_list[token_index]
            c_output0_silu = c_output0 / (1 + ops.exp(-c_output0))
            mul_output = c_output0_silu * self.columnparallerlinear_output1_list[token_index]
            row_weight_grad = self.matmul_grad_transpose_a(input_grad, mul_output)
        if row_weight.shape != row_weight_grad.shape:
            raise ValueError("Shape of the weights and grads in RowLayer are not the same!!!"
                             f"weights.shape={row_weight.shape} and grads.shape={row_weight_grad.shape}")
        # Mul Grad
        c_output0_silu_grad = self.columnparallerlinear_output1_list[token_index] * m_output_grad
        if not self.use_pipe_expert_recompute:
            c_output1_grad = self.columnparallerlinear_output0_silu_list[token_index] * m_output_grad
        else:
            c_output1_grad = c_output0_silu * m_output_grad
        # Silu Grad
        c_output0_grad = \
            silu_grad(self.columnparallerlinear_output0_list[token_index]) * c_output0_silu_grad
        c_output_grad = ops.cat((c_output0_grad, c_output1_grad), axis=-1)

        # ColumnParallerLinear Grad
        output_grad = self.matmul_notranspose(c_output_grad, col_weight)
        col_weight_grad = self.matmul_grad_transpose_a(
            c_output_grad,
            self.columnparallerlinear_input_list[token_index]
        )
        if col_weight.shape != col_weight_grad.shape:
            raise ValueError(f"Shape of the weights and grads in RowLayer are not the same!!!"
                             f"weights.shape={col_weight.shape} and grads.shape={col_weight_grad.shape}")
        return row_weight_grad, col_weight_grad, output_grad


    def swap_bprop_comp(self, token_index, input_grad, col_weight, row_weight):
        """computing expert gradients in backward process when using swap"""
        with ms.hal.StreamCtx(self.swap_stream):
            c_output0 = self.columnparallerlinear_output0_list[token_index].move_to("Ascend")
            c_output1 = self.columnparallerlinear_output1_list[token_index].move_to("Ascend")
        # RowParallerLinear Grad
        m_output_grad = self.matmul_notranspose(input_grad, row_weight)
        row_weight_grad = self.matmul_grad_transpose_a(
            input_grad,
            self.mul_output_list[token_index]
        )
        if row_weight.shape != row_weight_grad.shape:
            raise ValueError("Shape of the weights and grads in RowLayer are not the same!!!"
                             f"weights.shape={row_weight.shape} and grads.shape={row_weight_grad.shape}")
        # Mul Grad
        current_stream = ms.hal.current_stream()
        current_stream.wait_stream(self.swap_stream)
        c_output0_silu_grad = c_output1 * m_output_grad
        c_output1_grad = self.columnparallerlinear_output0_silu_list[token_index] * m_output_grad

        # Silu Grad
        c_output0_grad = silu_grad(c_output0) * c_output0_silu_grad
        c_output_grad = ops.cat((c_output0_grad, c_output1_grad), axis=-1)

        # ColumnParallerLinear Grad
        output_grad = self.matmul_notranspose(c_output_grad, col_weight)
        col_weight_grad = self.matmul_grad_transpose_a(
            c_output_grad,
            self.columnparallerlinear_input_list[token_index]
        )
        if col_weight.shape != col_weight_grad.shape:
            raise ValueError(f"Shape of the weights and grads in RowLayer are not the same!!!"
                             f"weights.shape={col_weight.shape} and grads.shape={col_weight_grad.shape}")
        return row_weight_grad, col_weight_grad, output_grad
