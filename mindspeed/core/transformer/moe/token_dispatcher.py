# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
from megatron.core import parallel_state, tensor_parallel
from mindspeed.core.transformer.moe.router import gather_from_sequence_parallel_region_to_moe_async


def token_permutation(self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind):
    self.hidden_shape = hidden_states.shape
    # [S/TP, B, H] -> [S*B/TP, H]
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

    # Permute the tokens across the expert parallel devices.
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        # [S*B/TP, H] -> [S*B, H]
        with torch.no_grad():
            global_indices, gi_handle = max_ind if isinstance(max_ind, tuple) else gather_from_sequence_parallel_region_to_moe_async(max_ind)
        global_probs, gp_handle = gather_from_sequence_parallel_region_to_moe_async(max_prob)
        global_hidden_states, ghs_handle = gather_from_sequence_parallel_region_to_moe_async(hidden_states)

        with torch.no_grad():
            gi_handle.wait()
            global_local_mask = (global_indices >= self.local_expert_indices[0]) & \
                                (global_indices <= self.local_expert_indices[-1])
            local_indices = global_indices.masked_select(global_local_mask)
            self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)
        self.global_local_map = global_local_mask.nonzero()[:, 0]

        if self.router_topk > 1:  # k > 1
            gp_handle.wait()
            self.local_probs = global_probs.masked_select(global_local_mask)
        else:
            self.local_probs = max_prob

        ghs_handle.wait()
        local_hidden_states = global_hidden_states[self.global_local_map, :]
    else:
        if self.router_topk > 1:
            global_local_mask = torch.ones_like(max_ind).bool()
            local_indices = max_ind.masked_select(global_local_mask)
            self.local_probs = max_prob.masked_select(global_local_mask)
            self.global_local_map = global_local_mask.nonzero()[:, 0]
            local_hidden_states = hidden_states[self.global_local_map, :]
        else:
            local_indices = max_ind
            self.local_probs = max_prob
            local_hidden_states = hidden_states
            self.global_local_map = None

        with torch.no_grad():
            # The indices of local_indices that give its sorted order along dim 0.
            self.indices = torch.argsort(local_indices, dim=0)
            tokens_per_expert = torch.histc(
                local_indices,
                bins=self.num_local_experts,
                min=self.local_expert_indices[0],
                max=self.local_expert_indices[-1],
            )
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

    permuted_local_hidden_states = local_hidden_states[self.indices, :]
    return (
        permuted_local_hidden_states,
        tokens_per_expert,
    )


def token_unpermutation(self, hidden_states: torch.Tensor, bias: torch.Tensor = None,):
    # Stage1: unpermute the tokens and bias locally respectively.w
    scores = self.local_probs.to(dtype=hidden_states.dtype)
    unpermuted_local_hidden = torch.zeros_like(hidden_states)
    unpermuted_local_hidden.index_put_((self.indices,), hidden_states[:self.indices.shape[0], :], accumulate=False)

    # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.
    if self.router_topk > 1:
        unpermuted_local_hidden = unpermuted_local_hidden * scores.view(-1, 1)

    unpermuted_local_bias = None
    if self.add_bias:
        assert bias is not None
        unpermuted_local_bias = torch.zeros_like(hidden_states)
        unpermuted_local_bias.index_put_((self.indices,), bias[:self.indices.shape[0], :], accumulate=False)

        if self.router_topk > 1:
            unpermuted_local_bias = unpermuted_local_bias * scores.view(-1, 1)

    output_total = unpermuted_local_hidden
    output_bias_total = unpermuted_local_bias

    # Unpermute the tokens across expert parallel devices.
    if self.config.sequence_parallel or (self.config.expert_model_parallel_size > 1):
        assert (
                self.global_local_map is not None
        ), "global_local_map is necessary for `AllGather`."
        ep_group_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # hidden_shape: [SeqLen/TP, MBS, HiddenSize], glboal_num_tokens = SeqLen/TP*MBS*(TP*EP)
        global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * ep_group_size
        global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
        unpermuted_global_hidden = torch.zeros(global_hidden_shape, dtype=hidden_states.dtype, device=torch.cuda.current_device())
        unpermuted_global_hidden.index_put_((self.global_local_map,),
                                            unpermuted_local_hidden[:self.global_local_map.shape[0], :],
                                            accumulate=True)

        output_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(unpermuted_global_hidden)
        if self.add_bias:
            # Unpermute the bias across expert parallel devices.
            unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
            unpermuted_global_bias.index_put_((self.global_local_map,),
                                              unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                              accumulate=True)

            output_bias_total = tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
                unpermuted_global_bias
            )
            # bias is duplicated across tensor parallelism ranks;
            # reduce scatter reduces bias across tensor parallel_ranks
            output_bias_total = (output_bias_total / parallel_state.get_tensor_model_parallel_world_size())
    else:
        if self.router_topk > 1:
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            unpermuted_global_hidden = torch.zeros(
                global_hidden_shape,
                dtype=hidden_states.dtype,
                device=torch.cuda.current_device(),
            )
            output_total = unpermuted_global_hidden.index_put((self.global_local_map,),
                                                              unpermuted_local_hidden[:self.global_local_map.shape[0], :],
                                                              accumulate=True)
            if self.add_bias:
                unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                output_bias_total = unpermuted_global_bias.index_put((self.global_local_map,),
                                                                     unpermuted_local_bias[:self.global_local_map.shape[0], :],
                                                                     accumulate=True)

    if self.router_topk == 1:
        output_total = output_total * scores
    output_total = output_total.view(self.hidden_shape)
    if self.add_bias:
        assert output_bias_total is not None
        if self.router_topk == 1:
            output_bias_total = output_bias_total * scores
        output_bias_total = output_bias_total.view(self.hidden_shape)
    else:
        output_bias_total = None

    return output_total, output_bias_total
