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
"""Sparse Attention APIs."""
import mindspore as ms
from mindspore import Tensor, nn, ops, mint, hal
from mindspore import dtype as mstype
from mindspore.ops import Send, Receive
import mindspore.communication.comm_func as comm_func

from mindspeed_ms.ops import SparseAttentionScore, SparseAttentionScoreGrad
from mindspeed_ms.core.parallel_state import is_hybrid_cp, get_context_parallel_group, \
    get_context_parallel_world_size, get_context_parallel_rank, get_sp_send_stream, \
    get_tensor_model_parallel_world_size, get_data_parallel_world_size, \
    get_context_parallel_group_for_hybrid_ring, get_context_parallel_for_hybrid_ring_world_size, \
    get_context_parallel_for_hybrid_ring_rank


class SparseAttention(nn.Cell):
    """Sequence parallelism with sparse attention.

    B -- Batch size
    S1 -- Sequence length of query. The value ranges from 1 to 32768 and is a multiple of 16.
    S2 -- Sequence length of key and value. The value ranges from 1 to 32768 and is a multiple of 16.
    N1 -- Num heads of query
    N2 -- Num heads of key and value, and N2 must be a factor of N1
    D -- Head size. Support value: 64, 80, 96, 120, 128 and 256.
    H1 -- Hidden size of query, which equals to N1 * D
    H2 -- Hidden size of key and value, which equals to N2 * D
    Args:
        head_num (int): The head num of query.
        keep_prob (float): The keep probability of dropout. Default: 1.0.
        scale_value (float): The scale factor of score. Default: 1.0.
        pre_tokens (int): Parameter for sparse computation, represents how many tokens are counted forward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        next_tokens (int): Parameter for sparse computation, represents how many tokens are counted backward.
        When sparse_mode is set to 1, 2, 3, or 5, this parameter does not take effect. Default: 2147483647.
        input_layout (str): Specifies the layout of input `query`, key and value. Default: "SBH".
        Currently only the value of "SBH" and "BNSD" are supported.
        sparse_mode (int): Indicates sparse mode. Default 0. Currently only sparse_mode = 0 is supported.

            - 0: Indicates the defaultMask mode. If attn_mask is not passed, the mask operation is not performed,
              and preTokens and nextTokens(internally assigned as INT_MAX) are ignored. If passed in, the full attn_mask
              matrix (S1 * S2) needs to be passed in, indicating that the part between preTokens and nextTokens needs to
              be calculated.
            - 1: Represents allMask, that is, passing in the complete attn_mask matrix.
            - 2: Representing the leftUpCausal mode corresponds to the lower triangle scenario divided by the left
              vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 3: Representing the rightDownCausal model corresponds to the lower triangle scene divided by the lower
              right vertex, and the optimized attn_mask matrix (2048*2048) is required.
            - 4: Represents the band scenario, that is, the part between counting preTokens and nextTokens, and the
              optimized attn_mask matrix (2048*2048) is required..
            - 5: Represents the prefix scenario, that is, on the basis of rightDownCasual, a matrix with length S1 and
              width N is added to the left side. The value of N is obtained by the new input prefix, and the N value of
              each Batch axis is different. Not implemented yet.
            - 6: Represents the global scenario, not implemented yet.
            - 7: Represents the dilated scenario, not implemented yet.
            - 8: Represents the block_local scenario, not implemented yet.
        use_attention_mask (bool): The value is True if attention_mask is passed. Default: False.
        use_alibi_mask (bool): The value is True if alibi_mask is passed. Default: False.
        Currently only use_alibi_mask = False is supported.
        use_mqa (bool): Specifies whether using MQA. Default: False. Currently only use_mqa = False is supported.
        attn_mask  (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048). Recommend to set attn_mask in order to speed up the SparseAttention call.

    Inputs:
        - **query** (Tensor[float16, bfloat16]) - The query tensor.
          Input tensor of shape :math:`(S1, B, H1)` or `(B, N1, S1, D)`.
        - **key** (Tensor[float16, bfloat16]) - The key tensor.
          Input tensor of shape :math:`(S2, B, H2)` or `(B, N2, S2, D)`.
        - **value** (Tensor[float16, bfloat16]) - The value tensor.
          Input tensor of shape :math:`(S2, B, H2)` or `(B, N2, S2, D)`.
        - **attn_mask** (Union[Tensor[uint8], None]) - The attention mask tensor. For each element, 0 indicates
          retention and 1 indicates discard. Input tensor of shape :math:`(B, N1, S1, S2)`, `(B, 1, S1, S2)`, `(S1, S2)`
          or (2048, 2048). If attn_mask = None, please use attn_mask_type to indicate the mask.
        - **alibi_mask** (Union[Tensor[float16, bfloat16], None]) - The position embedding code. If S is greater than
          1024 and the mask of the lower triangle is used, enter only the inverse 1024 lines of the lower triangle for
          memory optimization. Currently only alibi_mask = None is supported.
          Input tensor of shape :math:`(B, N1, S1, S2)`, `(1, N1, S1, S2)`, `(B, N1, 1024, S2)`, `(1, N1, 1024, S2)`
          or (1024, 1024).
        - **padding_mask** (None) - Reserved parameter. Not implemented yet.
          Currently only padding_mask = None is supported.
        - **prefix** (Union[Tensor[int64], None]) - N value of each Batch in the prefix sparse calculation scenario.
          Not implemented yet. Input tensor of shape :math:`(B,)`. Currently only prefix = None is supported.
        - **attn_mask_type** (str) - The attention mask type. Currently only value of None is supported.

    Outputs:
        - **attention_out** (Tensor[float16, bfloat16]) - The output of attention, its shape, and data type
          are the same as the query.

    Supported Platforms:
        ``Ascend910B``

    """

    def __init__(self,
                 head_num,
                 keep_prob=1.0,
                 scale_value=1.0,
                 pre_tokens=2147483647,
                 next_tokens=2147483647,
                 input_layout="SBH",
                 sparse_mode=0,
                 use_attention_mask=False,
                 use_alibi_mask=False,
                 use_mqa=False,
                 attn_mask=None
                 ):
        super(SparseAttention, self).__init__()
        self.head_num = head_num
        self.keep_prob = keep_prob
        self.scale_value = scale_value
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.use_attention_mask = use_attention_mask
        self.use_alibi_mask = use_alibi_mask
        self.use_mqa = use_mqa
        self.attn_mask = attn_mask
        self.dp = get_data_parallel_world_size()
        self.mp = get_tensor_model_parallel_world_size()
        if not is_hybrid_cp():
            self.sp = get_context_parallel_world_size()
        else:
            self.sp = get_context_parallel_for_hybrid_ring_world_size()

        if sparse_mode != 0:
            raise ValueError(f"Only sparse_mode = 0 is supported")

        if input_layout == "BSH":
            self.seq_dim = 1
            self.batch_dim = 0
        elif input_layout == "BNSD":
            self.seq_dim = 2
            self.batch_dim = 0
        elif input_layout == "SBH":
            self.seq_dim = 0
            self.batch_dim = 1
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if parallel_mode not in (
                ms.ParallelMode.STAND_ALONE, ms.ParallelMode.DATA_PARALLEL):
            raise ValueError(f"The sparse-attention only supports stand_alone and data_parallel,"
                             f"but got the paralle mode of {parallel_mode}")

        if self.use_alibi_mask:
            raise ValueError(f"Only use_alibi_mask = False is supported")

        if self.use_mqa:
            raise ValueError(f"Only use_mqa = False is supported")

        if self.attn_mask is not None:
            self.sparse_fa = self.prepare_sparse_attention()
            self.sparse_fa_grad = self.prepare_sparse_attention_grad()
            self.cp_skip_step = self.get_cp_skip_step()

        if self.sp > 1:
            self.stream_send = get_sp_send_stream()
            self.stream_recv = get_sp_send_stream()

    def p2p_communicate(self, rank, send_tensor, send_dst,
                        recv_src,
                        sp_group):
        """Point-to-point communications of KV and dKV in sparse attention"""

        stream_send = self.stream_send
        stream_recv = self.stream_recv
        send_recv_ops = []
        send_op = Send(0, send_dst, group=sp_group)
        send_op.add_prim_attr("dtype", send_tensor.dtype)
        recv_op = Receive(
            0,
            recv_src,
            shape=send_tensor.shape,
            dtype=send_tensor.dtype,
            group=sp_group)

        if rank % 2 == 0:
            with ms.hal.StreamCtx(stream_send):
                send_op(send_tensor)
            with ms.hal.StreamCtx(stream_recv):
                recv_tensor = recv_op(Tensor(0.0, dtype=send_tensor.dtype))
            send_recv_ops.append(stream_send)
            send_recv_ops.append(stream_recv)
        else:
            with ms.hal.StreamCtx(stream_recv):
                recv_tensor = recv_op(Tensor(0.0, dtype=send_tensor.dtype))
            with ms.hal.StreamCtx(stream_send):
                send_op(send_tensor)

            send_recv_ops.append(stream_recv)
            send_recv_ops.append(stream_send)
        send_recv_reqs = send_recv_ops
        return send_recv_reqs, recv_tensor

    def forward_update(self, prev_attn_out, prev_softmax_max, prev_softmax_sum,
                       cur_attn_out, cur_softmax_max, cur_softmax_sum):
        '''Update sparse attention output'''

        softmax_max = mint.maximum(prev_softmax_max, cur_softmax_max)
        prev_scale = mint.exp(prev_softmax_max - softmax_max)
        cur_scale = mint.exp(cur_softmax_max - softmax_max)

        prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
        cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
        softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

        prev_out_scale = prev_softmax_sum_scaled / softmax_sum
        cur_out_scale = cur_softmax_sum_scaled / softmax_sum

        n = prev_out_scale.shape[1]

        if self.input_layout == "BSH" or self.input_layout == "SBH":
            h = prev_attn_out.shape[-1]
            d = h // n
        elif self.input_layout == "BNSD":
            h = prev_attn_out.shape[1]
            d = prev_attn_out.shape[-1]
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        if self.input_layout == "BSH":
            prev_out_scale = ops.transpose(prev_out_scale, (0, 2, 1, 3))
            prev_out_scale = prev_out_scale.reshape(
                prev_out_scale.shape[0], prev_out_scale.shape[1], -1)
        elif self.input_layout == "SBH":
            prev_out_scale = ops.transpose(prev_out_scale, (2, 0, 1, 3))
            prev_out_scale = prev_out_scale.reshape(
                prev_out_scale.shape[0], prev_out_scale.shape[1], -1)
        elif self.input_layout == "BNSD":
            pass
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).tile((1, 1, 1, d))
        if self.input_layout == "BSH":
            cur_out_scale = ops.transpose(cur_out_scale, (0, 2, 1, 3))
            cur_out_scale = cur_out_scale.reshape(
                cur_out_scale.shape[0], cur_out_scale.shape[1], -1)
        elif self.input_layout == "SBH":
            cur_out_scale = ops.transpose(cur_out_scale, (2, 0, 1, 3))
            cur_out_scale = cur_out_scale.reshape(
                cur_out_scale.shape[0], cur_out_scale.shape[1], -1)
        elif self.input_layout == "BNSD":
            pass
        else:
            raise ValueError(
                f"Only input_layout = 'BSH' or 'BNSD' or 'SBH' is supported")

        attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
        return attn_out, softmax_max, softmax_sum

    def check_parameter(self, q, k, v, alibi_mask,
                        prefix, padding_mask, attn_mask_type):
        '''check sparse attention input'''

        if attn_mask_type is not None:
            raise ValueError(f"Only attn_mask_type = None is supported")
        if alibi_mask is not None:
            raise ValueError(f"Only alibi_mask = None is supported")
        if prefix is not None:
            raise ValueError(f"Only prefix = None is supported")
        if padding_mask is not None:
            raise ValueError(f"Only padding_mask = None is supported")

        s1 = q.shape[self.seq_dim]
        s2 = k.shape[self.seq_dim]
        s3 = v.shape[self.seq_dim]
        if s2 != s3:
            raise ValueError(
                f"The sequence length of input k and v should be equal, but got {s2} and {s3}")
        if s2 % 2 != 0:
            raise ValueError(
                f"The sequence length of input k and v must be an even number, but got {s2} and {s3}")

        b1 = q.shape[self.batch_dim]
        b2 = k.shape[self.batch_dim]
        b3 = v.shape[self.batch_dim]
        if b2 != b1:
            raise ValueError(
                f"The batch size of input q is not equal to k, but got batch size {b1} and {b2}")
        if b3 != b1:
            raise ValueError(
                f"The batch size of input q is not equal to v, but got batch_size {b1} and {b3}")

        if self.pre_tokens < s1 or self.pre_tokens < s2:
            raise ValueError(f"The pre_tokens should be larger or equal to the sequence of q and k,"
                             f"but got pre_tokens is {self.pre_tokens}, and the sequence length of q is {s1}"
                             f"and sequence length of kv is {s2}")

        if self.next_tokens < s1 or self.next_tokens < s2:
            raise ValueError(f"The next_tokens should be larger or equal to the sequence of q and k,"
                             f"but got next_tokens is {self.next_tokens}, and the sequence length of q is {s1}"
                             f"and sequence length of kv is {s2}")

        if self.next_tokens < 0:
            raise ValueError(
                f"The next_tokens should be larger or equal to 0, but got {self.next_tokens}")

    def prepare_qkv(self, q, k, v, send_kv, i):
        '''prepare qkv content'''
        if i == 0:
            cur_k, cur_v = k, v
        else:
            cur_k, cur_v = send_kv[0], send_kv[1]

        cur_attn_mask = None
        cur_q = q
        return cur_q, cur_k, cur_v, cur_attn_mask

    def prepare_flash_attention_grad_input(self, softmax_max, softmax_sum, q, attn_out, dout,
                                           cur_k, cur_v):
        '''prepare the flash attention grad input'''
        cur_attn_mask = None
        cur_q = q
        cur_softmax_max = softmax_max
        cur_softmax_sum = softmax_sum
        cur_attn_out = attn_out
        cur_dout = dout
        return cur_q, cur_k, cur_v, cur_dout, cur_attn_mask, cur_softmax_max, cur_softmax_sum, cur_attn_out

    def backward_update(self, cur_dq, cur_dk, cur_dv, dq, dk, dv, recv_dkv, recv_kv_dkv,
                        i, cp_size):
        '''Update the gradient during the backward pass'''
        if i == 0:
            hal.current_stream().wait_stream(self.stream_send)
            dq = cur_dq
            dk = cur_dk
            dv = cur_dv
        else:
            hal.current_stream().wait_stream(self.stream_send)

            if i == cp_size - 1:
                dkv = recv_dkv
            else:
                send_kv_dkv = recv_kv_dkv
                dkv = send_kv_dkv[1]

            dk, dv = dkv[0], dkv[1]
            dq = dq.add(cur_dq)
            dk = dk.add(cur_dk)
            dv = dv.add(cur_dv)
        return dq, dk, dv

    def get_cp_parallel_data(self):
        '''get context parallel groups'''
        in_hybrid_cp_mode = True
        if not is_hybrid_cp():
            in_hybrid_cp_mode = False
        if not in_hybrid_cp_mode:
            sp_group = get_context_parallel_group()
            cp_size = get_context_parallel_world_size()
            rank = get_context_parallel_rank()
        else:
            sp_group = get_context_parallel_group_for_hybrid_ring()
            cp_size = get_context_parallel_for_hybrid_ring_world_size()
            rank = get_context_parallel_for_hybrid_ring_rank()
        return sp_group, cp_size, rank

    def prepare_sparse_attention(self):
        '''prepare sparse attention'''
        fa_list = []
        _, cp_size, rank = self.get_cp_parallel_data()
        for i in range(cp_size):
            fa_list.append(
                SparseAttentionScore(head_num=self.head_num // self.mp,
                                     keep_prob=self.keep_prob,
                                     scale_value=self.scale_value,
                                     pre_tokens=self.pre_tokens,
                                     next_tokens=self.next_tokens,
                                     input_layout=self.input_layout,
                                     inner_precise=0,
                                     sparse_mode=self.sparse_mode,
                                     attn_mask=self.compile_mask_with_sparse_attention(
                                         self.attn_mask, i, rank, cp_size))
            )
        return fa_list

    def prepare_sparse_attention_grad(self):
        '''prepare sparse attention grad'''
        _, cp_size, rank = self.get_cp_parallel_data()
        fa_grad_list = []
        for i in range(cp_size):
            fa_grad_list.append(
                SparseAttentionScoreGrad(head_num=self.head_num // self.mp,
                                         keep_prob=self.keep_prob,
                                         scale_value=self.scale_value,
                                         pre_tokens=self.pre_tokens,
                                         next_tokens=self.next_tokens,
                                         input_layout=self.input_layout,
                                         inner_precise=0,
                                         sparse_mode=self.sparse_mode,
                                         atten_mask=self.compile_mask_with_sparse_attention_grad(
                                             self.attn_mask, i, rank, cp_size),
                                         blockpos_list=self.sparse_fa[cp_size - 1 - i].blockpos_list)
            )
        return fa_grad_list

    def compile_mask_with_sparse_attention(self, attn_mask, i, rank, cp_size):
        sub_seq_len = attn_mask.shape[-1] // cp_size
        start_index = ((rank - i) % cp_size) * sub_seq_len
        start_indices = [0] * (len(attn_mask.shape) - 1)
        start_indices.append(start_index)
        slice_shape = list(attn_mask.shape[:-1])
        slice_shape.append(sub_seq_len)
        cur_attn_mask = ops.slice(attn_mask, tuple(start_indices), tuple(slice_shape))
        return cur_attn_mask

    def compile_mask_with_sparse_attention_grad(self, attn_mask, i, rank, cp_size):
        sub_seq_len = attn_mask.shape[-1] // cp_size
        start_index = ((rank + i + 1) % cp_size) * sub_seq_len
        start_indices = [0] * (len(attn_mask.shape) - 1)
        start_indices.append(start_index)
        slice_shape = list(attn_mask.shape[:-1])
        slice_shape.append(sub_seq_len)
        cur_attn_mask = ops.slice(attn_mask, tuple(
            start_indices), tuple(slice_shape))
        return cur_attn_mask

    def get_cp_skip_step(self):
        '''Get the number of skip steps for SparseAttention'''
        sp_group, cp_size, _ = self.get_cp_parallel_data()
        skip_flag_all = mint.zeros((cp_size,), dtype=mstype.int8)
        self.skip_loop_flag = []
        for i in range(cp_size):
            skip_flag = mint.all(self.sparse_fa[i].attn_mask)
            skip_flag_all[i] = skip_flag
            self.skip_loop_flag.append(skip_flag.item())
        skip_flag_reduce_res = comm_func.all_reduce(skip_flag_all, group=sp_group)[0]
        cp_skip_step = 0
        for i in range(skip_flag_reduce_res.shape[0] - 1, -1, -1):
            if skip_flag_reduce_res[i] != cp_size:
                cp_skip_step = cp_size - 1 - i
                break
        return cp_skip_step

    def construct(self, q, k, v, attn_mask=None, alibi_mask=None, prefix=None,
                  padding_mask=None, attn_mask_type=None):
        '''Forward of SparseAttention block'''

        self.check_parameter(
            q,
            k,
            v,
            alibi_mask,
            prefix,
            padding_mask,
            attn_mask_type)

        if self.attn_mask is None:
            if attn_mask is None:
                raise ValueError("The input attn_mask should not be None, but found attn_mask = None.")
            self.attn_mask = attn_mask
            self.sparse_fa = self.prepare_sparse_attention()
            self.sparse_fa_grad = self.prepare_sparse_attention_grad()
            self.cp_skip_step = self.get_cp_skip_step()

        sp_group, cp_size, rank = self.get_cp_parallel_data()
        send_dst = (rank + 1) % cp_size
        recv_src = (rank + cp_size - 1) % cp_size

        send_kv = mint.cat((k.unsqueeze(0), v.unsqueeze(0)), 0)
        recv_tensor = None
        send_recv_ops = []
        softmax_max, softmax_sum = None, None
        attn_out = mint.zeros_like(q)
        cp_size = cp_size - self.cp_skip_step
        for i in range(cp_size):
            if send_recv_ops:
                hal.current_stream().wait_stream(self.stream_send)
                send_kv = recv_tensor

            if i < cp_size - 1:
                self.stream_send.wait_stream(hal.current_stream())
                send_recv_ops, recv_tensor = self.p2p_communicate(rank, send_kv, send_dst,
                                                                  recv_src, sp_group)
            cur_q, cur_k, cur_v, _ = self.prepare_qkv(q, k, v, send_kv, i)

            if self.skip_loop_flag[i]:
                continue

            drop_mask = None
            all_att_outs = self.sparse_fa[i](
                cur_q, cur_k, cur_v, alibi_mask, drop_mask, padding_mask, None, prefix)
            cur_attn_out = all_att_outs[3]
            cur_softmax_max = all_att_outs[0]
            cur_softmax_sum = all_att_outs[1]

            if softmax_max is None or softmax_sum is None:
                attn_out = cur_attn_out
                softmax_max = cur_softmax_max
                softmax_sum = cur_softmax_sum
            else:
                attn_out_updated, softmax_max_updated, softmax_sum_updated = self.forward_update(
                    attn_out, softmax_max, softmax_sum,
                    cur_attn_out, cur_softmax_max, cur_softmax_sum
                )
                attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

        self.softmax_max = softmax_max
        self.softmax_sum = softmax_sum

        self.k = send_kv[0]
        self.v = send_kv[1]
        attn_out = ops.cast(attn_out, self.k.dtype)

        return attn_out

    def bprop(self, q, k, v, attn_mask, attn_out, dout, attn_mask_type=None):
        '''Backward of SparseAttention block'''

        if attn_mask_type is not None:
            raise ValueError(f"Only attn_mask_type = None is supported.")

        if self.attn_mask is None:
            if attn_mask is None:
                raise ValueError("The input attn_mask should not be None, but found attn_mask = None.")
            self.attn_mask = attn_mask
            self.sparse_fa = self.prepare_sparse_attention()
            self.sparse_fa_grad = self.prepare_sparse_attention_grad()
            self.cp_skip_step = self.get_cp_skip_step()

        softmax_max = self.softmax_max
        softmax_sum = self.softmax_sum

        k = self.k
        v = self.v

        sp_group, cp_size, rank = self.get_cp_parallel_data()

        send_dst = (rank + cp_size - 1) % cp_size
        recv_src = (rank + 1) % cp_size


        kv = mint.cat((k.unsqueeze(0), v.unsqueeze(0)), 0)

        send_kv_dkv = mint.zeros((2, *kv.shape), dtype=kv.dtype)
        recv_kv_dkv = None
        recv_kv = None
        recv_dkv = None
        send_recv_ops = []
        dq = mint.zeros_like(q)
        dk = mint.zeros_like(k)
        dv = mint.zeros_like(v)

        cp_size = cp_size - self.cp_skip_step
        for i in range(cp_size):

            if send_recv_ops:
                if i == 1:
                    send_kv = recv_kv
                    send_kv_dkv[0] = send_kv

                else:
                    send_kv_dkv = recv_kv_dkv

            if i > 0:
                dkv = mint.cat((dk.unsqueeze(0), dv.unsqueeze(0)), 0)
                send_kv_dkv[1] = dkv
            self.stream_send.wait_stream(hal.current_stream())
            if i == 0:
                send_kv = kv
                send_recv_ops, recv_kv = self.p2p_communicate(rank, send_kv, send_dst,
                                                              recv_src, sp_group)
                cur_k, cur_v = k, v
            elif i == cp_size - 1:
                send_dkv = dkv
                send_recv_ops, recv_dkv = self.p2p_communicate(rank, send_dkv, send_dst,
                                                               recv_src, sp_group)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]
            else:
                send_recv_ops, recv_kv_dkv = self.p2p_communicate(rank, send_kv_dkv, send_dst,
                                                                  recv_src, sp_group)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]

            cur_q, cur_k, cur_v, cur_dout, _, cur_softmax_max, cur_softmax_sum, cur_attn_out = \
                self.prepare_flash_attention_grad_input(softmax_max, softmax_sum, q, attn_out, dout,
                                                        cur_k, cur_v)

            attn_grad_outs = self.sparse_fa_grad[i + self.cp_skip_step](
                cur_q, cur_k, cur_v, cur_dout, pse_shift=None, drop_mask=None, padding_mask=None,
                atten_mask=None, softmax_max=cur_softmax_max,
                softmax_sum=cur_softmax_sum, attention_in=cur_attn_out)

            cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

            dq, dk, dv = self.backward_update(
                cur_dq, cur_dk, cur_dv, dq, dk, dv, recv_dkv, recv_kv_dkv, i, cp_size)

        return dq, dk, dv, mint.zeros_like(attn_mask)
