# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
from einops import rearrange
from megatron.core import mpu


def _laser_attn_forward(qkvn, attn_mask=None, softmax_scale=1.):
    q, k, v, n = qkvn
    pre_tokens = k.shape[0]
    q, k, v = [rearrange(x, 's b n d -> b n s d').contiguous() for x in (q, k, v)]
    output = torch_npu.ascend_laser_attention(
        q, k, v,
        atten_mask=attn_mask if q.shape[2] == k.shape[2] else None,
        alibi_mask=None,
        scale=softmax_scale,
        head_num=n,
        input_layout="BNSD",
        keep_prob=1.0,
        pre_tokens=pre_tokens,
        next_tokens=1,
    )
    return output


def _laser_attn_backward(qkvn, dy, softmax_log_max_sum, atten_out, attn_mask=None, softmax_scale=1.):
    q, k, v, n = qkvn
    pre_tokens = k.shape[0]
    q, k, v, dy, atten_out = [rearrange(x, 's b n d -> b n s d').contiguous() for x in (q, k, v, dy, atten_out)]
    output = torch_npu.ascend_laser_attention_grad(
        attention_score_grad=dy,
        query=q,
        key=k,
        value=v,
        softmax_log_max_sum=softmax_log_max_sum,
        attention_score=atten_out,
        atten_mask=attn_mask if q.shape[2] == k.shape[2] else None,
        alibi_mask=None,
        scale=softmax_scale,
        head_num=q.shape[1],
        input_layout="BNSD",
        pre_tokens=pre_tokens,
        next_tokens=1,
    )
    return output


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


def forward_update(prev_attn_out, prev_softmax_log_max_sum,
                   cur_attn_out, cur_softmax_log_max_sum):
    if prev_attn_out is None:
        return cur_attn_out, cur_softmax_log_max_sum
    softmax_log_max_sum = torch.log(torch.exp(cur_softmax_log_max_sum) + torch.exp(prev_softmax_log_max_sum))
    attn_out = torch.exp(prev_softmax_log_max_sum - softmax_log_max_sum) * prev_attn_out + torch.exp(
        cur_softmax_log_max_sum - softmax_log_max_sum) * cur_attn_out
    return attn_out, softmax_log_max_sum


class AttentionWithCp(torch.autograd.Function):
    """Attention implementation with context parallelism"""

    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None):
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
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, n, d] -> [2, s, b, n, d]
            q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]

        send_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, n, d]
        recv_kv = None
        send_recv_ops = []
        attn_out, softmax_log_max_sum = None, None
        # (seed, offset, numels) for dropout mask
        # rng_states = [[0, 0, 0] for _ in range(cp_size)]

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
                cur_k, cur_v = k, v  # [2, s, b, n, d]
            else:
                cur_k, cur_v = send_kv[0], send_kv[1]  # [2, s, b, n, d]
            if causal:
                if i <= rank:
                    # [2, s, b, n, d] -> [2s, b, n, d]
                    cur_q = q.view(-1, *q.shape[2:])
                    if i == 0:
                        cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
                    else:
                        # only k[0] v[0] need to be calculated
                        cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
                else:
                    # only q[1] need to be calculated
                    cur_q = q[1]
                    # [2, s, b, n, d] -> [2s, b, n, d]
                    cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]

                # laser attention forward
                attn_outs = _laser_attn_forward(
                    (cur_q, cur_k, cur_v, n),
                    attn_mask=attn_mask,
                    softmax_scale=softmax_scale,
                )

                cur_attn_out, cur_softmax_log_max_sum = attn_outs[0], attn_outs[1]  # bnsd | bns1
                cur_softmax_log_max_sum = cur_softmax_log_max_sum.unsqueeze(3)
                cur_attn_out = cur_attn_out.to(torch.float)

                # (seed, offset, numels)
                # rng_states[i] = (attn_outs[4], attn_outs[5], attn_outs[6])

                if i == 0:
                    attn_out = cur_attn_out
                    softmax_log_max_sum = cur_softmax_log_max_sum  # bns1
                elif i <= rank:
                    attn_out_updated, softmax_log_max_sum_updated = forward_update(
                        attn_out, softmax_log_max_sum,
                        cur_attn_out, cur_softmax_log_max_sum
                    )
                    attn_out, softmax_log_max_sum = attn_out_updated, softmax_log_max_sum_updated
                else:
                    # [b, n, 2s, d] -> [b, n, 2, s, d]
                    attn_out = attn_out.view(*attn_out.shape[:2], 2, attn_out.shape[2] // 2, attn_out.shape[-1])
                    # [b, n, 2s, 1] -> [b, n, 2, s, 1]
                    softmax_log_max_sum = softmax_log_max_sum.view(*softmax_log_max_sum.shape[:2], 2,
                                                                   softmax_log_max_sum.shape[2] // 2,
                                                                   softmax_log_max_sum.shape[-1])
                    # [b, n, s, d], [b, n, s, 1], [b, n, s, d], [b, n, s, 1] -> [b, n, s, d], [b, n, s, 1]
                    attn_out_updated, softmax_log_max_sum_updated = forward_update(
                        attn_out[:, :, 1], softmax_log_max_sum[:, :, 1], cur_attn_out, cur_softmax_log_max_sum
                    )
                    attn_out[:, :, 1].copy_(attn_out_updated)  # [b, n, 2s, d]
                    softmax_log_max_sum[:, :, 1].copy_(softmax_log_max_sum_updated)  # [b, n, 2s, 1]

                    # [b, n, 2, s, d] -> [b, n, 2s, d]
                    attn_out = attn_out.view(*attn_out.shape[:2], -1, attn_out.shape[-1])
                    # [b, n, 2, s, 1] -> [b, n, 2s, 1]
                    softmax_log_max_sum = softmax_log_max_sum.view(*softmax_log_max_sum.shape[:2], -1,
                                                                   softmax_log_max_sum.shape[-1])
            else:
                attn_outs = _laser_attn_forward((q, cur_k, cur_v, n),
                                                attn_mask=attn_mask, softmax_scale=softmax_scale)

                cur_attn_out, cur_softmax_log_max_sum = attn_outs[0], attn_outs[1]
                cur_softmax_log_max_sum = cur_softmax_log_max_sum.unsqueeze(3)
                # rng_states[i] = (attn_outs[4], attn_outs[5], attn_outs[6])
                if i == 0:
                    attn_out = cur_attn_out
                    softmax_log_max_sum = cur_softmax_log_max_sum
                else:
                    attn_out_updated, softmax_log_max_sum_updated = forward_update(
                        attn_out, softmax_log_max_sum,
                        cur_attn_out, cur_softmax_log_max_sum
                    )
                    attn_out, softmax_log_max_sum = attn_out_updated, softmax_log_max_sum_updated

        k, v = send_kv[0], send_kv[1]
        if causal:
            q, k, v = [x.view(-1, *x.shape[2:]) for x in [q, k, v]]
        attn_out = rearrange(attn_out, 'b n s d -> s b (n d)').contiguous().to(q.dtype)
        ctx.save_for_backward(q, k, v, attn_mask, attn_out, softmax_log_max_sum)
        ctx.n = n
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.cp_global_ranks = cp_global_ranks
        ctx.use_cp_send_recv_overlap = use_cp_send_recv_overlap
        ctx.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
        return attn_out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, attn_mask, attn_out, softmax_log_max_sum = ctx.saved_tensors
        n = ctx.n
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        rank = ctx.cp_rank
        # Reversed order of forward
        send_dst = ctx.cp_global_ranks[(rank + cp_size - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
        use_cp_send_recv_overlap = ctx.use_cp_send_recv_overlap
        cp_group_for_send_recv_overlap = ctx.cp_group_for_send_recv_overlap

        #  [s, b, h] -> [s, b, n, d]
        attn_out = attn_out.view(*attn_out.shape[:2], n, attn_out.shape[-1] // n)
        dout = dout.view(*dout.shape[:2], n, dout.shape[-1] // n)

        if causal:
            # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
            q, k, v, attn_out, dout = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v, attn_out, dout]]
            # [b, n, 2s, 1] -> [b, n, 2, s, 1]
            softmax_log_max_sum = softmax_log_max_sum.view(softmax_log_max_sum.shape[0], softmax_log_max_sum.shape[1],
                                                           2, softmax_log_max_sum.shape[2] // 2,
                                                           softmax_log_max_sum.shape[-1])
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)  # [2, 2, s, b, h]
        send_kv_dkv = torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device)  # [2, 2, 2, s, b, h]
        recv_kv_dkv = None
        recv_kv = None
        recv_dkv = None
        send_recv_ops = []
        dq = torch.zeros_like(q)  # [2, s, b, n, d]
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        for i in range(cp_size):
            # wait until KV is received from recv_src
            if len(send_recv_ops) > 0:
                for send_recv_op in send_recv_ops:
                    send_recv_op.wait()
                if i == 1:  # only received kv in the second loop
                    send_kv = recv_kv
                    send_kv_dkv[0].copy_(send_kv)
                else:
                    send_kv_dkv = recv_kv_dkv
            if i > 0:
                dkv = torch.cat((dk.unsqueeze(0), dv.unsqueeze(0)), dim=0)
                send_kv_dkv[1].copy_(dkv)
            if i == 0:  # just send-recv kv in the first loop
                send_kv = kv
                recv_kv = torch.empty_like(send_kv)
                send_recv_ops = flash_attn_p2p_communicate(rank, send_kv, send_dst,
                                                           recv_kv, recv_src, cp_group, use_cp_send_recv_overlap,
                                                           cp_group_for_send_recv_overlap)
                cur_k, cur_v = k, v
            elif i == cp_size - 1:  # just send-recv dkv in the last loop
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
                    # [b, n, 2, s, 1] -> [b, n, 2s, 1]
                    cur_softmax_log_max_sum = softmax_log_max_sum.view(softmax_log_max_sum.shape[0],
                                                                       softmax_log_max_sum.shape[1], -1,
                                                                       softmax_log_max_sum.shape[-1])
                    # [2, s, b, n, d] -> [2s, b, n, d]
                    cur_q, cur_attn_out, cur_dout = [x.view(-1, *x.shape[2:]) for x in [q, attn_out, dout]]
                    if i == cp_size - 1:
                        cur_attn_mask = attn_mask
                        # [2, s, b, n, d] -> [2s, b, n, d]
                        cur_k, cur_v, = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
                    else:
                        cur_k, cur_v = [x[0] for x in [cur_k, cur_v]]
                else:
                    # [2, s, b, n, d] -> [2s, b, n, d]
                    cur_k, cur_v = [x.view(-1, *x.shape[2:]) for x in [cur_k, cur_v]]
                    # only q[1] attn_out[1] and dout[1] need to be calculated
                    cur_q, cur_attn_out, cur_dout = [x[1] for x in [q, attn_out, dout]]
                    # [b, n, 2, s, 1] -> [b, n, s, 1]
                    cur_softmax_log_max_sum = softmax_log_max_sum[:, :, 1].contiguous()

                # laser attention backward
                attn_grad_outs = _laser_attn_backward(
                    (cur_q, cur_k, cur_v, n),
                    cur_dout, cur_softmax_log_max_sum, cur_attn_out,
                    attn_mask=attn_mask, softmax_scale=softmax_scale)

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
                cur_dq, cur_dk, cur_dv = [rearrange(x, 'b n s d -> s b n d ').contiguous()
                                          for x in (cur_dq, cur_dk, cur_dv)]
                if i == 0:
                    if rank == cp_size - 1:
                        cur_dq = cur_dq.view(dq.shape)  # [2s, b, n, d] -> [2, s, b, n, d]
                        dq = cur_dq
                        dk[0].copy_(cur_dk)
                        dv[0].copy_(cur_dv)
                    else:
                        dq[1].copy_(cur_dq)
                        cur_dk = cur_dk.view(dk.shape)  # [2s, b, n, d] -> [2, s, b, n, d]
                        cur_dv = cur_dv.view(dv.shape)
                        dk = cur_dk
                        dv = cur_dv
                else:
                    # wait until dKV is received from recv_src
                    for send_recv_op in send_recv_ops:
                        send_recv_op.wait()
                    if i == cp_size - 1:  # only received dkv in the last loop
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
                        cur_dk = cur_dk.view(dk.shape)  # [2s, b, n, d] -> [2, s, b, n, d]
                        cur_dv = cur_dv.view(dv.shape)
                        dk.add_(cur_dk)
                        dv.add_(cur_dv)
            else:
                attn_grad_outs = _laser_attn_backward((q, cur_k, cur_v, n),
                                                      dout, softmax_log_max_sum, attn_out,
                                                      attn_mask=attn_mask, softmax_scale=softmax_scale)

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
                cur_dq, cur_dk, cur_dv = [rearrange(x, 'b n s d -> s b n d ').contiguous()
                                          for x in (cur_dq, cur_dk, cur_dv)]
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

        # [2, s, b, n, d] -> [2s, b, n, d]
        if causal:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None


def ringattn_context_parallel_la(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None):
    if attn_mask is None:
        attn_mask = torch.tril(torch.ones((2048, 2048), device=q.device, dtype=torch.uint8))
        attn_mask = (attn_mask == 0)
    out = AttentionWithCp.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask
    )
    return out
