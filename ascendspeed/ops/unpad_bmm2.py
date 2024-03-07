import os
import json
import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
import random
import numpy as np
from ascendspeed.op_builder import UnpadStridedBatchMatmulOpBuilder

class UnpadBmm2Function(torch.autograd.Function):

    ascendspeed_ops = UnpadStridedBatchMatmulOpBuilder().load()

    @staticmethod
    def forward(ctx, a, b, seqlen, head_num):
        """strided_batch_matmul bmm2 forward function
        
        Args:
            a(tensor): float16, The input left matrix, shape(headNum*seq_len[0]*seq_len[0]+...+headNum*seq_len[batch-1]*seq_len[batch-1])
            b(tensor): float16, The input right matrix, shape(sum(seq_len),hidden_size)
            seqlen(list): int32, shape param
            head_num: int32, shape param

        Return:
            c(tensor): float16, The output matrix, shape(sum(m*n)*head_num)
        """
        hidden_size = b.shape[1]
        batch = len(seqlen)
        ctx.seqlen = seqlen
        ctx.head_num = head_num
        ctx.hidden_size = hidden_size
        ctx.batch = batch
        ctx.save_for_backward(a, b)

        head_size = hidden_size // head_num
        seqlen_squared = [x**2 for x in seqlen]

        transA = 0
        transB = 0
        m = seqlen
        k = seqlen
        n =  [head_size] * batch
        lda = seqlen
        ldb = [hidden_size] * batch
        ldc =  [hidden_size] * batch
        strideA = seqlen_squared
        strideB =  [head_size] * batch
        strideC =  [head_size] * batch

        c = UnpadBmm2Function.ascendspeed_ops.npu_unpad_strided_batch_matmul(a, b, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num).view(sum(seqlen), hidden_size)

        return c

    @staticmethod
    def backward(ctx, grad_out):
        """strided_batch_matmul backward function, bmm2_grad1 and bmm2_grad2
        
        Args:
            grad_out(tensor): float16, left matrix of bmm2_grad1, right matrix of bmm2_grad2
            x(tensor): float16, The input left matrix of forward 
            y(tensor): float16, The input right matrix of forward 
            seqlen(list): int32, shape param
            head_num: int32, shape param
            hidden_size: int32, shape param
            batch: int32, len of seqlen

        Return:
            grad_x(tensor): float16, The output matrix of bmm2_grad1, shape(sum(m*n)*head_num)
            grad_y_ops(tensor): float16, The output matrix of bmm2_grad2, shape(sum(m*n)*head_num)
        """
        x, y = ctx.saved_tensors
        seqlen = ctx.seqlen
        head_num = ctx.head_num
        hidden_size = ctx.hidden_size
        batch = ctx.batch

        head_size = hidden_size // head_num
        seqlen_squared = [x**2 for x in seqlen]

        # bmm2_grad1
        transA = 0
        transB = 1
        m = seqlen
        k = [head_size] * batch
        n =  seqlen
        lda =  [hidden_size] * batch
        ldb = [hidden_size] * batch
        ldc =  seqlen
        strideA = [head_size] * batch
        strideB =  [head_size] * batch
        strideC =  seqlen_squared

        grad_x = UnpadBmm2Function.ascendspeed_ops.npu_unpad_strided_batch_matmul(grad_out, y, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num)

        # bmm2_grad2
        transA = 1
        transB = 0
        m = seqlen
        k = seqlen
        n = [head_size] * batch
        lda = seqlen
        ldb = [hidden_size] * batch
        ldc = [hidden_size] * batch
        strideA = seqlen_squared
        strideB = [head_size] * batch
        strideC = [head_size] * batch

        grad_y = UnpadBmm2Function.ascendspeed_ops.npu_unpad_strided_batch_matmul(x, grad_out, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, head_num).view(sum(seqlen), hidden_size)

        return grad_x, grad_y, None, None


class UnpadBmm2(Module):
    def __init__(self, head_num):
        super(UnpadBmm2, self).__init__()
        self.head_num = head_num

    def forward(self, a, b, seqlen):
        return UnpadBmm2Function.apply(a, b, seqlen, self.head_num)
