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
"""Ulysses Context Parallel test"""
import random
import math
import time
import logging
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, value_and_grad
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops.operations.nn_ops import FlashAttentionScore

from mindspeed_ms.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from utils import set_parallel_context

logging.basicConfig(level=logging.INFO)


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class FlashSelfAttention(ms.nn.Cell):
    """Implement the scaled dot product attention with softmax."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, head_num=None, seq_len=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.head_num = head_num
        self.seq_len = seq_len

        self.flash_attention = FlashAttentionScore(head_num=self.head_num,
                                                   keep_prob=1.,
                                                   scale_value=self.softmax_scale,
                                                   pre_tokens=2147483647,
                                                   next_tokens=0,
                                                   input_layout='SBH',
                                                   inner_precise=0,
                                                   sparse_mode=0)

    def construct(self, q, k, v, attention_mask):
        """Implement multihead softmax attention."""
        real_shift = None  # realShiftOptional
        drop_mask = None  # dropMaskOptional, 因为 keep_prob=1.0
        padding_mask = None  # paddingMaskOptional
        prefix = None  # prefixOptional

        # 调用 flash_attention，按照正确的参数顺序
        outputs = self.flash_attention(
            q,  # query
            k,  # key
            v,  # value
            real_shift,  # realShiftOptional
            drop_mask,  # dropMaskOptional
            padding_mask,  # paddingMaskOptional
            attention_mask,  # attenMaskOptional
            prefix,  # prefixOptional
        )

        # 假设 output 是返回的最后一个张量
        output = outputs[-1]
        return output


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """Slice data along sequence dimension into multiple chunks."""
    old_seq_len = data.shape[dim]
    new_seq_len = old_seq_len // cp_size
    assert dim == 0
    data = data[new_seq_len * cp_rank:new_seq_len * (cp_rank + 1)]
    return data


def run_ulysses_cp(cp_size, bs, seq_len, dtype):
    """
        Run the Ulysses context-parallel attention test.

        This function calculates the context-parallel and full attention outputs for comparison.
        It performs gradient computations and checks if the outputs and gradients within acceptable tolerance levels.

        Args:
            cp_size (int): Context parallel size, specifying the number of parallel groups.
            bs (int): Batch size for the attention computation.
            seq_len (int): Sequence length for the input tensors.
            dtype (mindspore.dtype): Data type to use for computations (e.g., mstype.bfloat16).

        Raises:
            AssertionError: If the context-parallel outputs or gradients deviate from the full attention results
            beyond the specified tolerance levels.

    """
    logging.info("Step 1: Setting random seed.")
    set_random_seed(1234)

    # 获取当前进程的 rank 和总进程数
    rank_id = get_rank()
    group_size = get_group_size()
    logging.info("Rank ID: %d, Group Size: %d", rank_id, group_size)

    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    ms_dtype = mstype.bfloat16

    attn_mask_np = ~np.tril(np.ones((seq_len, seq_len), dtype=bool))
    attn_mask_ms = Tensor(attn_mask_np, dtype=ms.bool_)

    q_ms = Tensor(np.random.randn(s, b, n * d), dtype=ms_dtype)
    k_ms = Tensor(np.random.randn(s, b, n * d), dtype=ms_dtype)
    v_ms = Tensor(np.random.randn(s, b, n * d), dtype=ms_dtype)

    logging.info("Step 2: Calculating full attention output.")
    # 计算全局输出（用于对比）
    core_attention_whole = FlashSelfAttention(causal=True, softmax_scale=scale, head_num=n, seq_len=s)
    grad_fn = value_and_grad(core_attention_whole, grad_position=(0, 1, 2))
    out_whole, grads_whole = grad_fn(q_ms, k_ms, v_ms, attn_mask_ms)
    kdgrad_whole, vdgrad_whole = grads_whole[1:]
    logging.info("Step 3: Full attention output calculated.")

    # 获取当前进程的数据切片
    q_ = get_data_on_this_cp_rank(q_ms, cp_size, rank_id)
    k_ = get_data_on_this_cp_rank(k_ms, cp_size, rank_id)
    v_ = get_data_on_this_cp_rank(v_ms, cp_size, rank_id)

    logging.info("Step 4: Calculating context-parallel attention output.")
    core_attention = FlashSelfAttention(causal=True, softmax_scale=scale, head_num=n // cp_size, seq_len=q_.shape[0])
    ulysses_attention = UlyssesContextAttention(core_attention, "cp-0-1")

    grad_fn = value_and_grad(ulysses_attention, grad_position=(0, 1, 2))
    time_taken = []
    for _ in range(10):
        start_time = time.time()
        out_, grads = grad_fn(q_, k_, v_, attn_mask_ms)
        out_.asnumpy()
        end_time = time.time()
        time_taken.append(end_time - start_time)
        logging.info(" -- Context-parallel attention output calculated. Time taken: %.4f seconds.",
                     end_time - start_time)

    kdgrad, vdgrad = grads[1:]
    # 聚合输出
    out_ulysses = ms.communication.comm_func.all_gather_into_tensor(out_, "cp-0-1")
    kdgrad_total = ms.communication.comm_func.all_gather_into_tensor(kdgrad, "cp-0-1")
    vdgrad_total = ms.communication.comm_func.all_gather_into_tensor(vdgrad, "cp-0-1")
    logging.info("Step 5: Gathered all outputs and gradients.")

    tols = {'atol': 5e-3, 'rtol': 5e-3}
    if dtype == ms.float16:
        tols = {'atol': 2.5e-2, 'rtol': 2.5e-2}

    out_is_close = ops.isclose(out_ulysses[0], out_whole, rtol=tols['rtol'], atol=tols['atol'])
    k_grad_is_close = ops.isclose(kdgrad_total[0], kdgrad_whole, rtol=tols['rtol'], atol=tols['atol'])
    v_grad_is_close = ops.isclose(vdgrad_total[0], vdgrad_whole, rtol=tols['rtol'], atol=tols['atol'])

    assert out_is_close.all(), "Output tensors are not close enough."
    assert k_grad_is_close.all(), "Key gradients are not close enough."
    assert v_grad_is_close.all(), "Value gradients are not close enough."
    assert min(time_taken) < 0.06, "Performance test failed."


class TestUlyssesCP:
    world_size = 2

    def test_ulysses_context_parallel_seq8192_bs2_bf16(self):
        set_parallel_context()
        logging.info("Testing Ulysses Context Parallel with sequence length 8192, batch size 2, and dtype bfloat16.")
        run_ulysses_cp(self.world_size, 2, 8192, mstype.bfloat16)
        logging.info("Test completed successfully.")


if __name__ == "__main__":
    # 初始化通信
    init()
    test_instance = TestUlyssesCP()
    test_instance.test_ulysses_context_parallel_seq8192_bs2_bf16()
