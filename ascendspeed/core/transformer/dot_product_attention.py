# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math
from functools import wraps

import torch
import torch_npu
from torch import Tensor
from megatron.training import get_args
from ascendspeed.core.context_parallel.ring_context_parallel import ringattn_context_parallel

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def dot_product_attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs['config']
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        fn(*args, **kwargs)
        config.context_parallel_size = cp_size

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if get_args().use_flash_attn:
            return dot_product_attention_forward(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


def dot_product_attention_forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    assert packed_seq_params is None
    args = get_args()

    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key = key.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
        )
        value = value.repeat_interleave(
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
        )

    query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]

    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

    if not hasattr(self, 'attention_mask'):
        seq_length = query.shape[0]
        self.attention_mask = (torch.tril(torch.ones([seq_length, seq_length]), diagonal=-(args.pre_tockens + 1))
                               + torch.triu(torch.ones([seq_length, seq_length]), diagonal=args.next_tockens + 1)).bool().npu()

    if args.context_parallel_size > 1 and args.context_parallel_algo == 'megatron_cp_algo':
        cp_para = dict()
        cp_para['causal'] = args.cp_attention_mask_type == 'causal'
        output = ringattn_context_parallel(query, key, value, self.num_attention_heads_per_partition, cp_para, scale, None)
    else:
        output = torch_npu.npu_fusion_attention(
            query, key, value, self.num_attention_heads_per_partition, 'SBH',
            pse=None,
            padding_mask=None,
            atten_mask=self.attention_mask,
            scale=scale,
            pre_tockens=args.pre_tockens,
            next_tockens=args.next_tockens,
            keep_prob=1 - self.attention_dropout.p,
            inner_precise=0
        )[0]

    return output
