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
"""Hybrid Context Parallel test."""
import random
import math
import logging
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, value_and_grad, mint
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_rank
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspeed_ms.core.context_parallel.ring_attention import RingAttention
from mindspeed_ms.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses, \
                                             get_context_parallel_group, initialize_model_parallel
from mindspeed_ms.training import parse_args
from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention

logging.basicConfig(level=logging.INFO)

def set_random_seed(seed):
    '''set random seed'''
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

class FlashSelfAttention(ms.nn.Cell):
    '''Flash Self Attention Net'''
    def __init__(self, head_num, input_layout, scale):
        super(FlashSelfAttention, self).__init__()

        self.flash_attention = FlashAttentionScore(head_num=head_num,
                                                   input_layout=input_layout,
                                                   scale_value=scale)

    def construct(self, *inputs):
        '''construct.'''
        output = self.flash_attention(*inputs)
        return output[-1]

def get_data_on_this_cp_rank(data, r_size, u_size, cp_rank, dim=0):
    '''get data on this cp rank'''
    cp_size = r_size * u_size
    args = get_args()
    if args.context_parallel_algo == "ulysses_cp_algo":
        data = data.chunk(cp_size, axis=dim)[cp_rank]
    elif args.context_parallel_algo == "megatron_cp_algo":
        data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim+1:])
        index = ms.tensor([cp_rank, (2 * cp_size - cp_rank - 1)])
        data = mint.index_select(data, dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim+2:])
    else:
        r_rank = cp_rank // u_size
        u_rank = cp_rank % u_size
        data = data.view(*data.shape[0:dim], 2 * r_size, data.shape[dim] // (2 * r_size), *data.shape[dim+1:])
        index = ms.tensor([r_rank, (2 * r_size - r_rank - 1)])
        data = mint.index_select(data, dim, index)
        data = data.view(*data.shape[0:dim], -1, *data.shape[dim+2:])
        data = data.chunk(u_size, axis=dim)[u_rank]
    return data

def get_data_on_this_cp_rank_general(data, r_size, u_size, cp_rank, dim=0):
    '''get data on this rank general'''
    cp_size = r_size * u_size
    data = data.chunk(cp_size, axis=dim)[cp_rank]
    return data

def get_attnmask_on_this_cp_rank_general(data, r_size, u_size, cp_rank, dim=0):
    '''get attnmask on this rank general'''
    args = get_args()
    attnmask = None
    if args.context_parallel_algo != "ulysses_cp_algo":
        attnmask = data.chunk(r_size, axis=dim)[cp_rank//u_size]
    return attnmask

def run_hybridattn_cp(args, dtype):
    '''Run hybrid cp test'''
    ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE,
                   deterministic='ON', pynative_synchronize=True)
    cp_size = args.context_parallel_size
    if args.context_parallel_algo == "ulysses_cp_algo":
        r_size = 1
        u_size = cp_size
    elif args.context_parallel_algo == "megatron_cp_algo":
        u_size = 1
        r_size = cp_size
    else:
        u_size = args.ulysses_degree_in_cp
        r_size = cp_size // u_size
    seq_len = args.seq_length
    logging.info("Step 1: Setting random seed.")
    set_random_seed(1234)

    logging.info("Step 2: initialize model parallel.")
    init()
    initialize_model_parallel(context_parallel_size=cp_size)
    logging.info("initialize model parallel succeeded!")

    rank_id = get_rank()

    b, n, s, d = 2, args.num_attention_heads, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    attn_mask_type = "causal" # causal user_defined
    attn_mask_np = ~np.tril(np.ones((seq_len, seq_len), dtype=bool))
    attn_mask_ms_whole = Tensor(attn_mask_np, dtype=ms.bool_)

    q_ms = Tensor(np.random.randn(s, b, n * d), dtype=dtype)
    k_ms = Tensor(np.random.randn(s, b, n * d), dtype=dtype)
    v_ms = Tensor(np.random.randn(s, b, n * d), dtype=dtype)

    logging.info("Step 3: Calculating full attention output.")
    core_attention_whole = FlashSelfAttention(head_num=n,
                                              input_layout='SBH',
                                              scale=scale)
    grad_fn = value_and_grad(core_attention_whole, grad_position=(0, 1, 2))
    out_whole, grads_whole = grad_fn(q_ms, k_ms, v_ms, None, None, None, attn_mask_ms_whole)
    kdgrad_whole, vdgrad_whole = grads_whole[1:]
    if attn_mask_type == "user_defined":
        out_whole_ = get_data_on_this_cp_rank_general(out_whole, r_size, u_size, rank_id)
        kdgrad_whole_ = get_data_on_this_cp_rank_general(kdgrad_whole, r_size, u_size, rank_id)
        vdgrad_whole_ = get_data_on_this_cp_rank_general(vdgrad_whole, r_size, u_size, rank_id)
    else:
        out_whole_ = get_data_on_this_cp_rank(out_whole, r_size, u_size, rank_id)
        kdgrad_whole_ = get_data_on_this_cp_rank(kdgrad_whole, r_size, u_size, rank_id)
        vdgrad_whole_ = get_data_on_this_cp_rank(vdgrad_whole, r_size, u_size, rank_id)

    logging.info("Step 4: Calculating context parallel attention output.")
    if attn_mask_type == "user_defined":
        q_ = get_data_on_this_cp_rank_general(q_ms, r_size, u_size, rank_id)
        k_ = get_data_on_this_cp_rank_general(k_ms, r_size, u_size, rank_id)
        v_ = get_data_on_this_cp_rank_general(v_ms, r_size, u_size, rank_id)
    else:
        q_ = get_data_on_this_cp_rank(q_ms, r_size, u_size, rank_id)
        k_ = get_data_on_this_cp_rank(k_ms, r_size, u_size, rank_id)
        v_ = get_data_on_this_cp_rank(v_ms, r_size, u_size, rank_id)

    attn_mask_ms_ = None
    if attn_mask_type == "user_defined" and args.context_parallel_algo != "ulysses_cp_algo":
        attn_mask_ms_ = get_attnmask_on_this_cp_rank_general(attn_mask_ms_whole, r_size, u_size, rank_id)

    if args.context_parallel_algo == "ulysses_cp_algo":
        core_attention = FlashSelfAttention(head_num=n//cp_size, input_layout='SBH', scale=scale)
        ulysses_attention = UlyssesContextAttention(core_attention, get_context_parallel_group())
        grad_fn = value_and_grad(ulysses_attention, grad_position=(0, 1, 2))
        out_, grads = grad_fn(q_, k_, v_, None, None, None, attn_mask_ms_whole)
    elif args.context_parallel_algo == "megatron_cp_algo":
        core_attention = RingAttention(head_num=n, input_layout='SBH', scale_value=scale)
        grad_fn = value_and_grad(core_attention, grad_position=(0, 1, 2))
        out_, grads = grad_fn(q_, k_, v_, attn_mask_ms_)
    else:
        core_attention = RingAttention(head_num=n//u_size, input_layout='SBH', scale_value=scale)
        ulysses_attention = UlyssesContextAttention(core_attention,
                                                    get_context_parallel_group_for_hybrid_ulysses())
        grad_fn = value_and_grad(ulysses_attention, grad_position=(0, 1, 2))
        out_, grads = grad_fn(q_, k_, v_, attn_mask_ms_)
    kdgrad_, vdgrad_ = grads[1:]

    logging.info("Step 5: context parallel attention output output calculated.")
    tols = {'atol': 5e-3, 'rtol': 5e-3}
    if dtype == ms.bfloat16:
        tols = {'atol': 2.5e-2, 'rtol': 2.5e-2}

    out_is_close = ops.isclose(out_, out_whole_, rtol=tols['rtol'], atol=tols['atol'])
    k_grad_is_close = ops.isclose(kdgrad_, kdgrad_whole_, rtol=tols['rtol'], atol=tols['atol'])
    v_grad_is_close = ops.isclose(vdgrad_, vdgrad_whole_, rtol=tols['rtol'], atol=tols['atol'])

    assert out_is_close.all(), "output tensors are not close enough"
    assert k_grad_is_close.all(), "kdgrad tensors are not close enough"
    assert v_grad_is_close.all(), "vdgrad tensors are not close enough"

class TestHybridCP:

    def test_hybrid_context_parallel_seq8192_bs2_bf16(self, args):
        logging.info("test_hybrid_context_parallel_seq8192_bs2_bf16")
        run_hybridattn_cp(args, mstype.bfloat16)
        logging.info("Test completed successfully.")

if __name__ == "__main__":
    config = parse_args()
    test_instance = TestHybridCP()
    test_instance.test_hybrid_context_parallel_seq8192_bs2_bf16(config)
