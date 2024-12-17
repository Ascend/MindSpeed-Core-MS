# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.

'''Ring Attention and Double Ring Attention with Context Parallelism'''
import os
import mindspore as ms
from mindspore import Tensor, nn, mint
from mindspore.mint.distributed import get_rank
from mindspeed_ms.ops.npu_fusion_attention import npu_fusion_attention, npu_fusion_attention_grad
from .utils import RingP2P, causal_out_update, general_out_update, forward_update

def causal_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_mask=None):
    '''Fetch the input of the forward pass'''

    cur_attn_mask = None
    if q_block_id == kv_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_attn_mask = attn_mask
        cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
    elif kv_block_id <= q_block_id:
        # [2, s, b, h] -> [2s, b, h]
        cur_q = q.view(-1, *q.shape[2:])
        # only k[0] v[0] need to be calculated
        cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # only q[1] need to be calculated
        cur_q = q[1]
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

    return cur_q, cur_k, cur_v, cur_attn_mask


def causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                          softmax_max, softmax_sum, attn_mask=None):
    '''Fetch the input of the backward pass'''

    cur_attn_mask = None
    if q_block_id >= kv_block_id:
        # [b, n, 2, s, 8] -> [b, n, 2s, 8]
        cur_softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1, softmax_max.shape[-1])
        cur_softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1, softmax_sum.shape[-1])
        # [2, s, b, h] -> [2s, b, h]
        cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
        if q_block_id == kv_block_id:
            cur_attn_mask = attn_mask
            # [2, s, b, h] -> [2s, b, h]
            cur_k, cur_v, = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        else:
            cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
    else:
        # [2, s, b, h] -> [2s, b, h]
        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
        # only q[1] attn_out[1] and dout[1] need to be calculated
        cur_q, cur_attn_out, cur_dout = [x[1] for x in [q, attn_out, dout]]
        cur_softmax_max, cur_softmax_sum = [x[:, :, 1, :, :] for x in [softmax_max, softmax_sum]]
    return cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask

def causal_grad_update(q_block_id, kv_block_id, cur_dq, cur_dk, cur_dv, dq, dk, dv):
    '''Update the gradient of q, k, v'''

    if q_block_id == kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        cur_dk = cur_dk.view(dk.shape)
        cur_dv = cur_dv.view(dv.shape)
        dq = mint.add(dq, cur_dq)
        dk = mint.add(dk, cur_dk)
        dv = mint.add(dv, cur_dv)
    elif q_block_id > kv_block_id:
        cur_dq = cur_dq.view(dq.shape)
        dq = mint.add(dq, cur_dq)
        dk[0] = mint.add(dk[0], cur_dk)
        dv[0] = mint.add(dv[0], cur_dv)
    else:
        dq[1] = mint.add(dq[1], cur_dq)
        cur_dk = cur_dk.view(dk.shape) # [2s, b, h] -> [2, s, b, h]
        cur_dv = cur_dv.view(dv.shape)
        dk = mint.add(dk, cur_dk)
        dv = mint.add(dv, cur_dv)

    return dq, dk, dv


def cal_row(cur_q, cur_k, cur_v, s, attn_info, cp_shape_order="SBH"):
    '''Calculate the output of the row'''

    # q: [s, b, h], kv: [2s, b, h]
    n, pse, attn_mask, softmax_scale, keep_prob, \
    q_index_list, kv_index_list = attn_info

    # r1c0
    cur_attn_mask = None
    attn_outs_r1c0 = npu_fusion_attention(
        cur_q, cur_k[:s], cur_v[:s], n, cp_shape_order,
        pse=pse,
        padding_mask=None,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else kv_index_list
        )
    # r1c1
    cur_attn_mask = attn_mask
    attn_outs_r1c1 = npu_fusion_attention(
        cur_q, cur_k[s:], cur_v[s:], n, cp_shape_order,
        pse=pse,
        padding_mask=None,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[1] * s,] if kv_index_list is not None else kv_index_list
    )

    # update row1
    attn_out = attn_outs_r1c0[0]
    softmax_max = attn_outs_r1c0[1]
    softmax_sum = attn_outs_r1c0[2]
    curr_attn_out = attn_outs_r1c1[0]
    curr_softmax_max = attn_outs_r1c1[1]
    curr_softmax_sum = attn_outs_r1c1[2]
    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(attn_out, softmax_max, softmax_sum,
                                                                                curr_attn_out, curr_softmax_max,
                                                                                curr_softmax_sum)
    return [attn_out_updated, softmax_max_updated, softmax_sum_updated]


def flash_attention_with_alibi_pse(q_block_id, kv_block_id, cur_qkv, attn_info, s,
                                   cp_shape_order="SBH"):
    '''Flash attention with alibi pse'''

    n, pse, cur_attn_mask, softmax_scale, keep_prob, \
    q_index_list, kv_index_list = attn_info
    cur_q, cur_k, cur_v = cur_qkv
    if q_block_id == kv_block_id:
        attn_outs_r0c0 = npu_fusion_attention(
            cur_q[:s], cur_k[:s], cur_v[:s], n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s,] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else None,
        )
        attn_outs_r1 = cal_row(cur_q[s:], cur_k, cur_v, s, attn_info, cp_shape_order)
        # get output
        attn_outs = []
        attn_outs.append(mint.cat([attn_outs_r0c0[0], attn_outs_r1[0]]))
        attn_outs.append(mint.cat([attn_outs_r0c0[1], attn_outs_r1[1]], dim=2))
        attn_outs.append(mint.cat([attn_outs_r0c0[2], attn_outs_r1[2]], dim=2))
    elif q_block_id > kv_block_id:
        attn_outs_r0c0 = npu_fusion_attention(
            cur_q[:s], cur_k, cur_v, n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s,] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else None,
        )
        attn_outs_r1c0 = npu_fusion_attention(
            cur_q[s:], cur_k, cur_v, n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else None,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else None,
        )
        # get output
        attn_outs = []
        attn_outs.append(mint.cat([attn_outs_r0c0[0], attn_outs_r1c0[0]]))
        attn_outs.append(mint.cat([attn_outs_r0c0[1], attn_outs_r1c0[1]], dim=2))
        attn_outs.append(mint.cat([attn_outs_r0c0[2], attn_outs_r1c0[2]], dim=2))
    else:
        attn_outs = cal_row(cur_q, cur_k, cur_v, s, attn_info, cp_shape_order)

    return attn_outs


def cal_row_grad(cur_q, cur_k, cur_v, cur_dout, cur_softmax_max,
                 cur_softmax_sum, cur_attn_out, attn_grad_info, s, kv_block_id, cp_shape_order="SBH"):
    '''Calculate the gradient of the row'''

    n, pse, attn_mask, softmax_scale, keep_prob, rng_states, \
    q_index_list, kv_index_list = attn_grad_info

    cur_attn_mask = None
    attn_grad_outs_r1c0 = npu_fusion_attention_grad(
        cur_q, cur_k[:s], cur_v[:s], cur_dout, n, cp_shape_order,
        pse=pse,
        padding_mask=None,
        softmax_max=cur_softmax_max,
        softmax_sum=cur_softmax_sum,
        attention_in=cur_attn_out,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        seed=rng_states[kv_block_id][0],
        offset=rng_states[kv_block_id][1],
        numels=rng_states[kv_block_id][2],
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else kv_index_list
    )

    cur_attn_mask = attn_mask
    attn_grad_outs_r1c1 = npu_fusion_attention_grad(
        cur_q, cur_k[s:], cur_v[s:], cur_dout, n, cp_shape_order,
        pse=pse,
        padding_mask=None,
        softmax_max=cur_softmax_max,
        softmax_sum=cur_softmax_sum,
        attention_in=cur_attn_out,
        atten_mask=cur_attn_mask,
        scale=softmax_scale,
        pre_tokens=s,
        next_tokens=0 if cur_attn_mask is not None else s,
        keep_prob=keep_prob,
        seed=rng_states[kv_block_id][0],
        offset=rng_states[kv_block_id][1],
        numels=rng_states[kv_block_id][2],
        sparse_mode=3 if cur_attn_mask is not None else 0,
        q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else q_index_list,
        kv_start_idx=[kv_index_list[1] * s,] if kv_index_list is not None else kv_index_list
    )

    return attn_grad_outs_r1c0, attn_grad_outs_r1c1


def flash_attention_with_alibi_pse_grad(q_block_id, kv_block_id, cur_qkv, cur_dout, cur_attn_out,
                                        cur_softmax_max, cur_softmax_sum, attn_grad_info, s,
                                        cp_shape_order="SBH"):
    '''Flash attention with alibi pse'''

    n, pse, cur_attn_mask, softmax_scale, keep_prob, rng_states, \
    q_index_list, kv_index_list = attn_grad_info
    cur_q, cur_k, cur_v = cur_qkv

    if q_block_id == kv_block_id:
        attn_grad_outs_r0c0 = npu_fusion_attention_grad(
            cur_q[:s], cur_k[:s], cur_v[:s], cur_dout[:s], n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, :s],
            softmax_sum=cur_softmax_sum[:, :, :s],
            attention_in=cur_attn_out[:s],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s,] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs_r1c0, attn_grad_outs_r1c1 = cal_row_grad(
            cur_q[s:], cur_k, cur_v, cur_dout[s:], cur_softmax_max[:, :, s:], cur_softmax_sum[:, :, s:],
            cur_attn_out[s:], attn_grad_info, s, kv_block_id, cp_shape_order
        )
        attn_grad_outs = []
        attn_grad_outs.append(mint.cat(
            [attn_grad_outs_r0c0[0], attn_grad_outs_r1c0[0] + attn_grad_outs_r1c1[0]]))
        attn_grad_outs.append(mint.cat(
            [attn_grad_outs_r0c0[1] + attn_grad_outs_r1c0[1], attn_grad_outs_r1c1[1]]))
        attn_grad_outs.append(mint.cat(
            [attn_grad_outs_r0c0[2] + attn_grad_outs_r1c0[2], attn_grad_outs_r1c1[2]]))

    elif q_block_id > kv_block_id:
        attn_grad_outs_r0c0 = npu_fusion_attention_grad(
            cur_q[:s], cur_k, cur_v, cur_dout[:s], n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, :s],
            softmax_sum=cur_softmax_sum[:, :, :s],
            attention_in=cur_attn_out[:s],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[0] * s,] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs_r1c0 = npu_fusion_attention_grad(
            cur_q[s:], cur_k, cur_v, cur_dout[s:], n, cp_shape_order,
            pse=pse,
            padding_mask=None,
            softmax_max=cur_softmax_max[:, :, s:],
            softmax_sum=cur_softmax_sum[:, :, s:],
            attention_in=cur_attn_out[s:],
            atten_mask=cur_attn_mask,
            scale=softmax_scale,
            pre_tokens=s,
            next_tokens=0 if cur_attn_mask is not None else s,
            keep_prob=keep_prob,
            seed=rng_states[kv_block_id][0],
            offset=rng_states[kv_block_id][1],
            numels=rng_states[kv_block_id][2],
            sparse_mode=3 if cur_attn_mask is not None else 0,
            q_start_idx=[q_index_list[1] * s,] if q_index_list is not None else q_index_list,
            kv_start_idx=[kv_index_list[0] * s,] if kv_index_list is not None else kv_index_list
        )
        attn_grad_outs = []
        attn_grad_outs.append(mint.cat([attn_grad_outs_r0c0[0], attn_grad_outs_r1c0[0]]))
        attn_grad_outs.append(attn_grad_outs_r0c0[1] + attn_grad_outs_r1c0[1])
        attn_grad_outs.append(attn_grad_outs_r0c0[2] + attn_grad_outs_r1c0[2])

    else:
        attn_grad_outs_r1c0, attn_grad_outs_r1c1 = cal_row_grad(
            cur_q, cur_k, cur_v, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_out,
            attn_grad_info, s, kv_block_id, cp_shape_order
        )
        attn_grad_outs = []
        attn_grad_outs.append(attn_grad_outs_r1c0[0] + attn_grad_outs_r1c1[0])
        attn_grad_outs.append(mint.cat([attn_grad_outs_r1c0[1], attn_grad_outs_r1c1[1]]))
        attn_grad_outs.append(mint.cat([attn_grad_outs_r1c0[2], attn_grad_outs_r1c1[2]]))


    return attn_grad_outs



# pylint:disable=R1705
class AttentionWithCp(nn.Cell):
    """Attention implementation with context parallelism"""
    def __init__(self):
        super().__init__()
        self.block_size = None
        self.batch_size = None

    def construct(self, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                  actual_seq_qlen=None, actual_seq_kvlen=None):
        '''Forward of RingAttention block'''

        #set block_size and batch size
        self.block_size = q.shape[0]
        self.batch_size = q.shape[1]
        keep_prob = 1. - dropout_p
        causal = cp_para['causal']
        cp_group = cp_para.get("cp_group")
        cp_size = cp_para.get("cp_size")
        rank = cp_para.get("rank")
        cp_global_ranks = cp_para.get("cp_global_ranks")
        cp_group_for_send_recv_overlap = cp_para.get("cp_group_for_send_recv_overlap")
        # WARNING: Degrade to original ring attention, if ranks and comm groups for double ring are not provided
        cp_inner_ranks = cp_para.get("cp_inner_ranks", [get_rank()])
        cp_outer_ranks = cp_para.get("cp_outer_ranks", cp_global_ranks)
        cp_group_for_intra_window = cp_para.get('cp_group_for_intra_window')
        cp_group_for_intra_window_send_recv_overlap = cp_para.get('cp_group_for_intra_window_send_recv_overlap')
        cp_shape_order = cp_para.get("cp_shape_order", "SBH")

        pse = cp_para.get("pse")
        inner_ring = RingP2P(cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap)
        outer_ring = RingP2P(cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap)
        inner_size = len(cp_inner_ranks)
        outer_size = cp_size // inner_size

        if softmax_scale is None:
            head_dim = q.shape[-1] // n
            softmax_scale = head_dim ** (-0.5)
        if causal and attn_mask is None:
            attn_mask_len = int(os.getenv("ATTN_MASK_LEN", default="2048"))
            attn_mask = mint.ones((attn_mask_len, attn_mask_len), dtype=ms.bool_)
            attn_mask = mint.triu(attn_mask, diagonal=1)
        if causal:
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
            q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        cur_kv = mint.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0) # [2, 2, s, b, h]
        next_kv = mint.ones_like(cur_kv)
        next_round_kv = mint.ones_like(cur_kv)
        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        rng_states = [[0, 0, 0] for _ in range(cp_size)]
        global_attn_outs = [attn_out, softmax_max, softmax_sum, rng_states]
        q_block_id, kv_block_id, kv_block_id_outer = rank, rank, rank

        for j in range(outer_size):
            kv_block_id = kv_block_id_outer
            kv_block_offset = (kv_block_id // inner_size) * inner_size
            if j < outer_size - 1:
                next_round_kv = outer_ring.async_send_recv(send_tensor=cur_kv, recv_tensor=next_round_kv)
            for i in range(inner_size):
                # wait until KV is received from recv_src
                if i < inner_size - 1:
                    next_kv = inner_ring.async_send_recv(send_tensor=cur_kv, recv_tensor=next_kv)

                cur_k, cur_v = cur_kv[0], cur_kv[1] # [2, s, b, h]
                if causal:
                    cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(q_block_id, kv_block_id,
                                                                              q, cur_k, cur_v, attn_mask)

                    # flash attention forward
                    if pse is None:
                        attn_outs = npu_fusion_attention(
                            cur_q, cur_k, cur_v, n, cp_shape_order,
                            pse=None,
                            padding_mask=None,
                            atten_mask=cur_attn_mask,
                            scale=softmax_scale,
                            pre_tokens=cur_k.shape[0],
                            next_tokens=0 if cur_attn_mask is not None else cur_k.shape[0],
                            keep_prob=keep_prob,
                            sparse_mode=3 if cur_attn_mask is not None else 0
                        )
                    else:
                        q_index_list = [q_block_id, cp_size * 2 - 1 - q_block_id]
                        kv_index_list = [kv_block_id, cp_size * 2 - 1 - kv_block_id]
                        attn_info = [n, pse, cur_attn_mask, softmax_scale, keep_prob,
                                     q_index_list, kv_index_list]
                        s = q.shape[1]
                        attn_outs = flash_attention_with_alibi_pse(
                            q_block_id, kv_block_id,
                            (cur_q, cur_k, cur_v),
                            attn_info,
                            s, cp_shape_order
                        )

                    global_attn_outs = causal_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs)

                else:
                    # [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
                    this_mask = self.compute_mask(
                        actual_seq_qlen, actual_seq_kvlen,
                        q_block_id, kv_block_id,
                        attn_mask
                    )

                    attn_outs = npu_fusion_attention(
                        q, cur_k, cur_v, n, cp_shape_order,
                        pse=None,
                        padding_mask=None,
                        atten_mask=this_mask,
                        scale=softmax_scale,
                        pre_tokens=cur_k.shape[0],
                        next_tokens=cur_k.shape[0],
                        keep_prob=keep_prob,
                        sparse_mode=1
                    )

                    global_attn_outs = general_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs)

                if inner_ring.wait():
                    cur_kv, next_kv = next_kv, cur_kv # double buffer
                    kv_block_id = (kv_block_id + inner_size - 1) % inner_size + kv_block_offset

            if outer_ring.wait():
                cur_kv, next_round_kv = next_round_kv, cur_kv # double buffer
                kv_block_id_outer = (kv_block_id_outer + cp_size - inner_size) % cp_size

        k, v = cur_kv[0], cur_kv[1]
        attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
        if causal:
            q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]

        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]

        self.attn_mask = attn_mask
        #save forward outputs
        self.softmax_max = softmax_max
        self.softmax_sum = softmax_sum
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.cp_rank = rank
        self.cp_global_ranks = cp_global_ranks
        self.cp_inner_ranks = cp_inner_ranks
        self.cp_outer_ranks = cp_outer_ranks
        self.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_global_ranks)
        self.kv_block_id = kv_block_id
        self.keep_prob = keep_prob
        self.rng_states = rng_states
        self.pse = pse
        self.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
        self.cp_group_for_intra_window = cp_group_for_intra_window
        self.cp_group_for_intra_window_send_recv_overlap = cp_group_for_intra_window_send_recv_overlap
        return attn_out

    # pylint: disable=W0613
    def bprop(self, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
              actual_seq_qlen=None, actual_seq_kvlen=None, attn_out=None, dout=None):

        '''Backward of RingAttention block'''
        #q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum = self.saved_tensors

        del dropout_p
        del actual_seq_qlen
        del actual_seq_kvlen
        softmax_max = self.softmax_max
        softmax_sum = self.softmax_sum
        attn_mask = self.attn_mask
        if len(attn_mask) == 1:
            attn_mask = attn_mask[0]

        causal = self.causal
        cp_group = self.cp_group
        cp_size = self.cp_size
        rank = self.cp_rank
        keep_prob = self.keep_prob
        rng_states = self.rng_states
        pse = self.pse
        cp_group_for_send_recv_overlap = self.cp_group_for_send_recv_overlap
        cp_group_for_intra_window = self.cp_group_for_intra_window
        cp_group_for_intra_window_send_recv_overlap = self.cp_group_for_intra_window_send_recv_overlap
        cp_shape_order = cp_para.get("cp_shape_order", "SBH")
        # Reversed order of forward
        inner_size = len(self.cp_inner_ranks)
        outer_size = len(self.cp_outer_ranks)

        intra_kv_comm = RingP2P(self.cp_inner_ranks, cp_group_for_intra_window,
                                cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        intra_dkv_comm = RingP2P(self.cp_inner_ranks, cp_group_for_intra_window,
                                 cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        inter_kv_comm = RingP2P(self.cp_outer_ranks, cp_group,
                                cp_group_for_send_recv_overlap, is_backward=True)
        inter_dkv_comm = RingP2P(self.cp_dkv_outer_ranks, cp_group,
                                 cp_group_for_send_recv_overlap, is_backward=True)


        if causal:
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
            q, k, v, attn_out, dout = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v, attn_out, dout]]
            # [b, n, 2s, 8] -> [b, n, 2, s, 8]
            softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                           2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
            softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                           2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])

        def backward_step_helper(q_block_id, kv_block_id, q, cur_k, cur_v):
            if causal:
                # flash attention backward
                step_inputs = causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v,
                                                    attn_out, dout, softmax_max, softmax_sum,
                                                    attn_mask=attn_mask)
                (cur_q, cur_k, cur_v, cur_attn_out,
                 cur_dout, cur_softmax_max, cur_softmax_sum,
                 cur_attn_mask
                 ) = step_inputs
                if pse is None:
                    attn_grad_outs = npu_fusion_attention_grad(
                        cur_q, cur_k, cur_v, cur_dout, n,
                        cp_shape_order,
                        pse=None,
                        padding_mask=None,
                        atten_mask=cur_attn_mask,
                        softmax_max=cur_softmax_max,
                        softmax_sum=cur_softmax_sum,
                        attention_in=cur_attn_out,
                        scale=softmax_scale,
                        pre_tokens=cur_k.shape[0],
                        next_tokens=0 if cur_attn_mask is not None else cur_k.shape[0],
                        sparse_mode=3 if cur_attn_mask is not None else 0,
                        keep_prob=keep_prob,
                        seed=rng_states[kv_block_id][0],
                        offset=rng_states[kv_block_id][1],
                        numels=rng_states[kv_block_id][2],
                    )
                else:
                    q_index_list = [q_block_id, cp_size * 2 - 1 - q_block_id]
                    kv_index_list = [kv_block_id, cp_size * 2 - 1 - kv_block_id]
                    attn_grad_info = [n, pse, cur_attn_mask, softmax_scale, keep_prob, rng_states,
                                      q_index_list, kv_index_list]
                    s = q.shape[1]
                    attn_grad_outs = flash_attention_with_alibi_pse_grad(
                        q_block_id, kv_block_id,
                        (cur_q, cur_k, cur_v), cur_dout, cur_attn_out,
                        cur_softmax_max, cur_softmax_sum,
                        attn_grad_info, s, cp_shape_order
                    )

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
            else:
                this_mask = self.compute_mask(
                    self.actual_seq_qlen, self.actual_seq_kvlen,
                    q_block_id, kv_block_id,
                    attn_mask
                )
                attn_grad_outs = npu_fusion_attention_grad(
                    q, cur_k, cur_v, dout, n,
                    cp_shape_order,
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    softmax_max=softmax_max,
                    softmax_sum=softmax_sum,
                    attention_in=attn_out,
                    scale=softmax_scale,
                    pre_tokens=cur_k.shape[0],
                    next_tokens=cur_k.shape[0],
                    sparse_mode=1,
                    keep_prob=keep_prob,
                    seed=rng_states[kv_block_id][0],
                    offset=rng_states[kv_block_id][1],
                    numels=rng_states[kv_block_id][2],
                )
                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]

            return cur_dq, cur_dk, cur_dv


        cur_kv_dkv = mint.zeros((2, 2, *k.shape), dtype=k.dtype) # [2, 2, 2, s, b, h]
        cur_kv_dkv[0].copy_(mint.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0))
        next_kv_dkv = cur_kv_dkv.copy()
        next_round_kv_dkv = cur_kv_dkv.copy()

        cur_kv, cur_dkv = cur_kv_dkv[0], cur_kv_dkv[1]
        next_kv, next_dkv = next_kv_dkv[0], next_kv_dkv[1]
        next_round_kv, next_round_dkv = next_round_kv_dkv[0], next_round_kv_dkv[1]

        q_block_id, kv_block_id, kv_block_id_outer = rank, self.kv_block_id, self.kv_block_id


        dq = mint.zeros_like(q)# [2, s, b, h]
        for j in range(outer_size):
            kv_block_id = kv_block_id_outer
            kv_block_offset = (kv_block_id // inner_size) * inner_size
            if j > 0:
                inter_kv_comm.wait()
                cur_kv, next_round_kv = next_round_kv, cur_kv

            if j + 1 != outer_size:
                next_round_kv = inter_kv_comm.async_send_recv(send_tensor=cur_kv, recv_tensor=next_round_kv)

            for i in range(inner_size):
                if i > 0:
                    intra_kv_comm.wait()
                    cur_kv, next_kv = next_kv, cur_kv

                if i + 1 != inner_size:
                    next_kv = intra_kv_comm.async_send_recv(send_tensor=cur_kv, recv_tensor=next_kv)

                cur_k, cur_v = cur_kv[0], cur_kv[1]

                dq_step, dk_step, dv_step = backward_step_helper(q_block_id, kv_block_id, q, cur_k, cur_v)

                if i == 0 and j > 0: # receive dk dv from last window
                    inter_dkv_comm.wait()
                    cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
                elif i > 0: # receive dk dv from last step
                    intra_dkv_comm.wait()
                    cur_dkv, next_dkv = next_dkv, cur_dkv

                dk, dv = cur_dkv[0], cur_dkv[1]
                # update qkv grades
                if causal:
                    dq, dk, dv = causal_grad_update(q_block_id, kv_block_id, dq_step, dk_step, dv_step, dq, dk, dv)
                else:
                    dq = mint.add(dq, dq_step)
                    dk = mint.add(dk, dk_step)
                    dv = mint.add(dv, dv_step)

                cur_dkv[0] = dk
                cur_dkv[1] = dv

                if i + 1 != inner_size:
                    next_dkv = intra_dkv_comm.async_send_recv(send_tensor=cur_dkv, recv_tensor=next_dkv)

                kv_block_id = (kv_block_id + 1) % inner_size + kv_block_offset

            if intra_dkv_comm.wait():
                cur_dkv, next_dkv = next_dkv, cur_dkv

            if j + 1 != outer_size:
                next_round_dkv = inter_dkv_comm.async_send_recv(send_tensor=cur_dkv, recv_tensor=next_round_dkv)

            kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_size

        if inter_dkv_comm.wait():
            cur_dkv, next_round_dkv = next_round_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]


        # [2, s, b, h] -> [2s, b, h]
        if causal:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None, None, None

    def compute_mask(self, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
        '''Compute mask for attention'''

        from bisect import bisect_right

        def batch_index(seq1d):
            seq_len = seq1d[-1] // self.batch_size
            end_points = list(range(seq_len, seq1d[-1] + 1, seq_len))
            indexes = [0] + [bisect_right(seq1d, p) for p in end_points]
            seq_batch = [seq1d[indexes[i]:indexes[i + 1]] for i in range(len(indexes) - 1)]
            return [[elem - i * seq_len for elem in seq] for i, seq in enumerate(seq_batch)]

        if actual_seq_qlen:
            actual_seq_qlen = batch_index(actual_seq_qlen)
            actual_seq_kvlen = batch_index(actual_seq_kvlen)
            block_size = self.block_size
            actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
            sub_seq_qlen = [Tensor(x[1:]) - Tensor(x[:-1]) for x in actual_seq_qlen]
            sub_seq_qid = mint.stack([mint.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]) # B S

            this_ids = sub_seq_qid[:, q_block_id * block_size:(q_block_id + 1) * block_size]
            this_tile = this_ids.unsqueeze(dim=2) # B S 1

            actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
            sub_seq_kvlen = [Tensor(x[1:]) - Tensor(x[:-1]) for x in actual_seq_kvlen]
            sub_seq_kvid = mint.stack([mint.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]) # B S
            other_ids = sub_seq_kvid[:, kv_block_id * block_size:(kv_block_id + 1) * block_size]
            other_tile = other_ids.unsqueeze(dim=1) # B 1 S

            mask = this_tile == other_tile # B S S
            if kv_block_id > q_block_id:
                mask = mint.zeros_like(mask)
            elif kv_block_id == q_block_id:
                mask = mint.tril(mask)

            return mint.logical_not(mask).unsqueeze(dim=1)  # B 1 S S
        else:
            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None
