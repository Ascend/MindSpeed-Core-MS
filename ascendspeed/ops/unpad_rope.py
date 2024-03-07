import os
import json
import torch
import torch_npu
import numpy as np
from ascendspeed.op_builder import UnpadRopeOpBuilder
torch.set_printoptions(profile="full")

class UnpadRopeFunc(torch.autograd.Function):

    ascendspeed_ops = UnpadRopeOpBuilder().load()

    @staticmethod
    def forward(ctx, q, k, cos, sin, seqlen):
        """rope forward function
        
        Args:
            q(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size_q）, hidden_size_q=head_num*head_size，hidden_size_q%hidden_size_k=0
            k(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size_k）
            cos(tensor): float16, The input matrix, shape（max_seq_len,head_size）
            sin(tensor): float16, The input matrix, shape（max_seq_len,head_size）
            seqlen(tensor): int32, shape param
            rotaryCoeff: int32
            cosFormat: int32

        Return:
            out_tensors[0](tensor): float16, The output matrix, shape（sum(seq_len),hidden_size_q）
            out_tensors[1](tensor): float16, The output matrix, shape（sum(seq_len),hidden_size_k）
        """
        rotaryCoeff=2
        cosFormat=1
        ctx.cos = cos
        ctx.sin = sin
        ctx.seqlen=seqlen
        qseqlen = torch.tensor(seqlen, device=q.device, dtype=torch.int32)

        out_tensors = UnpadRopeFunc.ascendspeed_ops.npu_unpad_rope(q, k, cos, sin, qseqlen, rotaryCoeff, cosFormat)
        return out_tensors[0],out_tensors[1]

    @staticmethod
    def backward(ctx, q_embed_grad,k_embed_grad):
        """rope backward function
        
        Args:
            q_embed_grad(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size）
            k_embed_grad(tensor): float16, The input matrix, shape（sum(seq_len),hidden_size）
            cos(tensor): float16, The input matrix, shape（max_seq_len,head_size）
            sin(tensor): float16, The input matrix, shape（max_seq_len,head_size）
            seqlen(list): int32, shape param

        Return:
            out_tensors[0](tensor): float16, The output matrix, shape（sum(seq_len),hidden_size）
            out_tensors[1](tensor): float16, The output matrix, shape（sum(seq_len),hidden_size）
        """
        out_tensors = UnpadRopeFunc.ascendspeed_ops.npu_unpad_rope_grad(q_embed_grad, k_embed_grad,
            ctx.cos, ctx.sin, ctx.seqlen)
        return out_tensors[0], out_tensors[1], None, None, None

class UnpadRope(torch.nn.Module):
    def __init__(self):
        super(UnpadRope, self).__init__()

    def forward(self, q, k, cos, sin, seqlen, offset=0):
        # 处理输入数据，调用Ropefunction
        q_embed, k_embed = UnpadRopeFunc.apply(q, k, cos, sin, seqlen)
        return q_embed,k_embed
