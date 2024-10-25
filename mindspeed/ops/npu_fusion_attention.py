# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
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

from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore.ops import functional as F
from mindspore.ops import DropoutGenMask
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype


def npu_fusion_attention(query, key, value, head_num, input_layout, *, 
                    pse=None, padding_mask=None, atten_mask=None, scale=1., 
                    keep_prob=1., pre_tokens=2147483647, next_tokens=2147483647, 
                    inner_precise=0, prefix=None, actual_seq_qlen=None, 
                    actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, 
                    sync=False, pse_type=1, q_start_idx=None, kv_start_idx=None):
    """
    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.
        - **softmax_max** (Tensor[float32]) - (B, N1, S1, 8) when input_layout is not `TND` else (T1, N1, D)
        - **softmax_sum** (Tensor[float32]) - (B, N1, S1, 8) when input_layout is not `TND` else (T1, N1, D)
    """
    fa_op = FlashAttentionScore(head_num=head_num,
                                scale_value=scale,
                                keep_prob=keep_prob,
                                pre_tokens=pre_tokens,
                                next_tokens=next_tokens,
                                inner_precise=inner_precise,
                                input_layout=input_layout,
                                sparse_mode=sparse_mode)
    
    # create drop_mask. 
    enable_dropout = keep_prob < 1.0
    drop_mask_bits = None

    if input_layout != "TND":
        if input_layout == "BSH":
            bsz, q_seq_len, _ = query.shape
            _, kv_seq_len, _ = key.shape
        elif input_layout == "SBH":
            q_seq_len, bsz, _ = query.shape
            kv_seq_len, _, _ = key.shape
        elif input_layout == "BNSD":
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape
        elif input_layout == "BSND":
            bsz, q_seq_len, _, _ = query.shape
            _, kv_seq_len, _, _ = key.shape
        else:
            raise ValueError(f"input_layout is invalid")
    
        if enable_dropout:
            drop_gen_mask = DropoutGenMask()
            keep_prob_tensor = Tensor(keep_prob, dtype=mstype.float16)
            drop_mask_bits = F.reshape(
                drop_gen_mask((bsz, head_num, q_seq_len, kv_seq_len), keep_prob_tensor),
                (bsz, head_num, q_seq_len, kv_seq_len // 8))

    out = fa_op(query, key, value,
                real_shift=pse,
                drop_mask=drop_mask_bits,
                padding_mask=padding_mask,
                attn_mask=atten_mask,
                prefix=prefix,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen)

    sfm_max, sfm_sum, sfm_out, atten_out = out

    return atten_out, sfm_max, sfm_sum