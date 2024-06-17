# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
from einops import rearrange
from megatron.core import mpu


def flash_attn_p2p_communicate(rank, send_tensor, send_dst,
                               recv_tensor, recv_src,
                               cp_group,
                               use_cp_send_recv_overlap,
                               cp_group_for_send_recv_overlap,
                               ):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if use_cp_send_recv_overlap:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group_for_send_recv_overlap)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group_for_send_recv_overlap)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(torch.distributed.irecv,
                                              recv_tensor,
                                              recv_src,
                                              cp_group)
            send_op = torch.distributed.P2POp(torch.distributed.isend,
                                              send_tensor,
                                              send_dst,
                                              cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)

    return send_recv_reqs


def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum):
    # update softmax_max
    origin_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)

    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled

    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum

    # [b, n, s, 8] -> [s, b, h]
    n = prev_out_scale.shape[1]
    h = prev_attn_out.shape[-1]
    d = h // n
    prev_out_scale = prev_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    prev_out_scale = rearrange(prev_out_scale, 'b n s d -> s b (n d)').contiguous()
    cur_out_scale = cur_out_scale[..., 0].unsqueeze(3).repeat(1, 1, 1, d)
    cur_out_scale = rearrange(cur_out_scale, 'b n s d -> s b (n d)').contiguous()

    # update output
    attn_out = prev_attn_out * prev_out_scale + cur_attn_out * cur_out_scale
    attn_out = attn_out.to(origin_dtype)
    return attn_out, softmax_max, softmax_sum


class AttentionWithCp(torch.autograd.Function):
    """Attention implementation with context parallelism"""

    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.):
        keep_prob = 1. - dropout_p
        causal = cp_para['causal']
        cp_group = cp_para.get("cp_group")
        cp_size = cp_para.get("cp_size")
        rank = cp_para.get("rank")
        cp_global_ranks = cp_para.get("cp_global_ranks")
        use_cp_send_recv_overlap = cp_para.get("use_cp_send_recv_overlap")
        cp_group_for_send_recv_overlap = cp_para.get("cp_group_for_send_recv_overlap")

        send_dst = cp_global_ranks[(rank + 1) % cp_size]
        recv_src = cp_global_ranks[(rank + cp_size - 1) % cp_size]

        if softmax_scale is None:
            head_dim = q.shape[-1] // n
            softmax_scale = head_dim ** (-0.5)
        if causal and attn_mask is None:
            attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        if causal:
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
            q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        send_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0) # [2, 2, s, b, h]
        recv_kv = None
        send_recv_ops = []
        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        rng_states = [[0, 0, 0] for _ in range(cp_size)]

        for i in range(cp_size):
            # wait until KV is received from recv_src
            if len(send_recv_ops) > 0:
                for send_recv_op in send_recv_ops:
                    send_recv_op.wait()
                send_kv = recv_kv
            if i < cp_size - 1:
                recv_kv = torch.empty_like(send_kv)
                send_recv_ops = flash_attn_p2p_communicate(rank, send_kv, send_dst,
                                                           recv_kv, recv_src, cp_group, use_cp_send_recv_overlap,
                                                           cp_group_for_send_recv_overlap)
            if i == 0:
                cur_k, cur_v = k, v
            else:
                cur_k, cur_v = send_kv[0], send_kv[1] # [2, s, b, h]
            if causal:
                cur_attn_mask = None
                if i == 0:
                    # [2, s, b, h] -> [2s, b, h]
                    cur_attn_mask = attn_mask
                    cur_q, cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [q, cur_k, cur_v]]
                elif i <= rank:
                    # [2, s, b, h] -> [2s, b, h]
                    cur_q = q.view(-1, *q.shape[2:])
                    # only k[0] v[0] need to be calculated
                    cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
                else:
                    # only q[1] need to be calculated
                    cur_q = q[1]
                    # [2, s, b, h] -> [2s, b, h]
                    cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

                # flash attention forward
                attn_outs = torch_npu.npu_fusion_attention(
                    cur_q, cur_k, cur_v, n, "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=cur_attn_mask,
                    scale=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
                    keep_prob=keep_prob,
                    sparse_mode=3 if cur_attn_mask is not None else 0
                )

                # if i <= rank: [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
                # else: [s, b, h], [b, n, s, 8], [b, n, s, 8]
                cur_attn_out = attn_outs[0] 
                cur_softmax_max = attn_outs[1]
                cur_softmax_sum = attn_outs[2]
                # (seed, offset, numels)
                rng_states[i] = (attn_outs[4], attn_outs[5], attn_outs[6])

                if i == 0:
                    attn_out = cur_attn_out
                    softmax_max = cur_softmax_max
                    softmax_sum = cur_softmax_sum
                elif i <= rank:
                    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                        attn_out, softmax_max, softmax_sum,
                        cur_attn_out, cur_softmax_max, cur_softmax_sum
                    )
                    attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated
                else:
                    # [2s, b, h] -> [2, s, b, h]
                    attn_out = attn_out.view(2, attn_out.shape[0] // 2, *attn_out.shape[1:])
                    # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                    softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                                   2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
                    softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                                   2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
                    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                        attn_out[1], softmax_max[:, :, 1, :, :], softmax_sum[:, :, 1, :, :],
                        cur_attn_out, cur_softmax_max, cur_softmax_sum
                    )
                    attn_out[1].copy_(attn_out_updated)
                    softmax_max[:, :, 1, :, :].copy_(softmax_max_updated)
                    softmax_sum[:, :, 1, :, :].copy_(softmax_sum_updated)
                    # [2, s, b, h] -> [2s, b, h]
                    attn_out = attn_out.view(-1, *attn_out.shape[2:])
                    # [b, n, 2, s, 8] -> [b, n, 2s, 8]
                    softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                                   softmax_max.shape[-1])
                    softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                                   softmax_sum.shape[-1])
            else:
                # [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
                this_mask = attn_mask[(rank - i) % cp_size] if isinstance(attn_mask, list) else None
                attn_outs = torch_npu.npu_fusion_attention(
                    q, cur_k, cur_v, n, "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    scale=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=cur_k.shape[0],
                    keep_prob=keep_prob,
                    sparse_mode=0
                )

                cur_attn_out, cur_softmax_max, cur_softmax_sum = attn_outs[0], attn_outs[1], attn_outs[2]
                rng_states[i] = (attn_outs[4], attn_outs[5], attn_outs[6])
                if i == 0:
                    attn_out = cur_attn_out
                    softmax_max = cur_softmax_max
                    softmax_sum = cur_softmax_sum
                else:
                    attn_out_updated, softmax_max_updated, softmax_sum_updated = forward_update(
                        attn_out, softmax_max, softmax_sum,
                        cur_attn_out, cur_softmax_max, cur_softmax_sum
                    )
                    attn_out, softmax_max, softmax_sum = attn_out_updated, softmax_max_updated, softmax_sum_updated

        k, v = send_kv[0], send_kv[1]
        if causal:
            q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]
        
        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]
        
        ctx.save_for_backward(q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.cp_global_ranks = cp_global_ranks
        ctx.keep_prob = keep_prob
        ctx.rng_states = rng_states
        ctx.use_cp_send_recv_overlap = use_cp_send_recv_overlap
        ctx.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, *attn_mask, attn_out, softmax_max, softmax_sum = ctx.saved_tensors
        if len(attn_mask) == 1:
            attn_mask = attn_mask[0]

        n = ctx.n
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        rank = ctx.cp_rank
        keep_prob = ctx.keep_prob
        rng_states = ctx.rng_states
        # Reversed order of forward
        send_dst = ctx.cp_global_ranks[(rank + cp_size - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
        use_cp_send_recv_overlap = ctx.use_cp_send_recv_overlap
        cp_group_for_send_recv_overlap = ctx.cp_group_for_send_recv_overlap

        if causal:
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
            q, k, v, attn_out, dout = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v, attn_out, dout]]
            # [b, n, 2s, 8] -> [b, n, 2, s, 8]
            softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                           2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
            softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                           2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0) # [2, 2, s, b, h]
        send_kv_dkv = torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device) # [2, 2, 2, s, b, h]
        recv_kv_dkv = None
        recv_kv = None
        recv_dkv = None
        send_recv_ops = []
        dq = torch.zeros_like(q) # [2, s, b, h]
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        for i in range(cp_size):
            # wait until KV is received from recv_src
            if len(send_recv_ops) > 0:
                for send_recv_op in send_recv_ops:
                    send_recv_op.wait()
                if i == 1: # only received kv in the second loop
                    send_kv = recv_kv
                    send_kv_dkv[0].copy_(send_kv)
                else:
                    send_kv_dkv = recv_kv_dkv
            if i > 0:
                dkv = torch.cat((dk.unsqueeze(0), dv.unsqueeze(0)), dim=0)
                send_kv_dkv[1].copy_(dkv)
            if i == 0: # just send-recv kv in the first loop
                send_kv = kv
                recv_kv = torch.empty_like(send_kv)
                send_recv_ops = flash_attn_p2p_communicate(rank, send_kv, send_dst,
                                                           recv_kv, recv_src, cp_group, use_cp_send_recv_overlap,
                                                           cp_group_for_send_recv_overlap)
                cur_k, cur_v = k, v
            elif i == cp_size - 1: # just send-recv dkv in the last loop
                send_dkv = send_kv_dkv[1]
                recv_dkv = torch.empty_like(send_dkv)
                send_recv_ops = flash_attn_p2p_communicate(rank, send_dkv, send_dst,
                                                           recv_dkv, recv_src, cp_group, use_cp_send_recv_overlap,
                                                           cp_group_for_send_recv_overlap)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]
            else:
                recv_kv_dkv = torch.empty_like(send_kv_dkv)
                send_recv_ops = flash_attn_p2p_communicate(rank, send_kv_dkv, send_dst,
                                                           recv_kv_dkv, recv_src, cp_group, use_cp_send_recv_overlap,
                                                           cp_group_for_send_recv_overlap)
                cur_k, cur_v = send_kv_dkv[0][0], send_kv_dkv[0][1]

            if causal:
                cur_attn_mask = None
                if i >= cp_size - rank - 1:
                    # [b, n, 2, s, 8] -> [b, n, 2s, 8]
                    cur_softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1], -1,
                                                       softmax_max.shape[-1])
                    cur_softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1], -1,
                                                       softmax_sum.shape[-1])
                    # [2, s, b, h] -> [2s, b, h]
                    cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
                    if i == cp_size - 1:
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

                # flash attention backward
                attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                    cur_q, cur_k, cur_v, cur_dout, n,
                    "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=cur_attn_mask,
                    softmax_max=cur_softmax_max,
                    softmax_sum=cur_softmax_sum,
                    attention_in=cur_attn_out,
                    scale_value=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
                    sparse_mode=3 if cur_attn_mask is not None else 0,
                    keep_prob=keep_prob,
                    seed=rng_states[cp_size - i - 1][0],
                    offset=rng_states[cp_size - i - 1][1],
                    numels=rng_states[cp_size - i - 1][2],
                )

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
                if i == 0:
                    if rank == cp_size - 1:
                        cur_dq = cur_dq.view(dq.shape) # [2s, b, h] -> [2, s, b, h]
                        dq = cur_dq
                        dk[0].copy_(cur_dk)
                        dv[0].copy_(cur_dv)
                    else:
                        dq[1].copy_(cur_dq)
                        cur_dk = cur_dk.view(dk.shape) # [2s, b, h] -> [2, s, b, h]
                        cur_dv = cur_dv.view(dv.shape)
                        dk = cur_dk
                        dv = cur_dv
                else:
                    # wait until dKV is received from recv_src
                    for send_recv_op in send_recv_ops:
                        send_recv_op.wait()
                    if i == cp_size - 1: # only received dkv in the last loop
                        dkv = recv_dkv
                    else:
                        send_kv_dkv = recv_kv_dkv
                        dkv = send_kv_dkv[1]
                    dk, dv = dkv[0], dkv[1]
                    if i >= cp_size - rank - 1:
                        if i == cp_size - 1:
                            cur_dq = cur_dq.view(dq.shape)
                            cur_dk = cur_dk.view(dk.shape)
                            cur_dv = cur_dv.view(dv.shape)
                            dq.add_(cur_dq)
                            dk.add_(cur_dk)
                            dv.add_(cur_dv)
                        else:
                            cur_dq = cur_dq.view(dq.shape)
                            dq.add_(cur_dq)
                            dk[0].add_(cur_dk)
                            dv[0].add_(cur_dv)
                    else:
                        dq[1].add_(cur_dq)
                        cur_dk = cur_dk.view(dk.shape) # [2s, b, h] -> [2, s, b, h]
                        cur_dv = cur_dv.view(dv.shape)
                        dk.add_(cur_dk)
                        dv.add_(cur_dv)
            else:
                this_mask = attn_mask[(rank + i + 1) % cp_size] if attn_mask else None
                attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                    q, cur_k, cur_v, dout, n,
                    "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    softmax_max=softmax_max,
                    softmax_sum=softmax_sum,
                    attention_in=attn_out,
                    scale_value=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=cur_k.shape[0],
                    sparse_mode=0,
                    keep_prob=keep_prob,
                    seed=rng_states[cp_size - i - 1][0],
                    offset=rng_states[cp_size - i - 1][1],
                    numels=rng_states[cp_size - i - 1][2],
                )
                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
                if i == 0:
                    dq = cur_dq
                    dk = cur_dk
                    dv = cur_dv
                else:
                    # wait until dKV is received from recv_src
                    for send_recv_op in send_recv_ops:
                        send_recv_op.wait()
                    # only received dkv in the last loop
                    if i == cp_size - 1:
                        dkv = recv_dkv
                    else:
                        send_kv_dkv = recv_kv_dkv
                        dkv = send_kv_dkv[1]
                    dk, dv = dkv[0], dkv[1]
                    dq.add_(cur_dq)
                    dk.add_(cur_dk)
                    dv.add_(cur_dv)

        # [2, s, b, h] -> [2s, b, h]
        if causal:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None


def ringattn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.):
    out = AttentionWithCp.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p
    )
    return out
