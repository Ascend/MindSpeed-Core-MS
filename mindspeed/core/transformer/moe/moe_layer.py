# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import types
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel


def moe_layer_init_wrapper(init_func):
    @wraps(init_func)
    def moe_layer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        global_args = get_args()
        self.moe_adaptive_recompute_activation = global_args.moe_adaptive_recompute_activation
        self.moe_adaptive_recompute_activation_scale = global_args.moe_adaptive_recompute_activation_scale
        self.recompute_threshold = parallel_state.get_tensor_model_parallel_world_size() * parallel_state.get_data_parallel_world_size() * \
            self.config.moe_router_topk * self.moe_adaptive_recompute_activation_scale / self.config.num_moe_experts
        self.token_dispatcher.all_tokens_per_expert = None
        self.forward = types.MethodType(moe_adaptive_forward, self)
    return moe_layer_init


def moe_adaptive_forward(self, hidden_states: torch.Tensor):
    def custom_forward(hidden_states):
        scores, indices = self.router(hidden_states)
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            hidden_states, scores, indices
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        return output, mlp_bias
   
    threshold = hidden_states.shape[0] * hidden_states.shape[1] * self.recompute_threshold
    moe_adaptive_recompute_activation_bool = self.moe_adaptive_recompute_activation and \
        (self.token_dispatcher.all_tokens_per_expert is None or torch.max(self.token_dispatcher.all_tokens_per_expert) > threshold)
    if self.moe_layer_recompute or moe_adaptive_recompute_activation_bool:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)
    return output, mlp_bias