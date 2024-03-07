import torch
import torch_npu
import numpy as np
import random

from ascendspeed.op_builder import UnpadStridedBatchMatmulOpBuilder

repeatTime=5

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].flatten(), golden_out_tensors[0].flatten(), rtol=0.001, atol=0.001)

def golden_calc1(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
    seq_lengths = m
    A = in_tensors[0].view(sum(seq_lengths), headNum, -1).permute(1, 0, 2).contiguous()
    B = in_tensors[1].view(sum(seq_lengths), headNum, -1).permute(1, 0, 2).contiguous()
    attention_scores_list = []
    start = 0
    for i, seq_length in enumerate(seq_lengths):
        end = start + seq_length
        attention_scores = torch.matmul(A[:, start:end, ...], B[:, start:end, ...].transpose(2, 1))
        attention_scores_list.append(attention_scores.flatten())
        start = end
    C = torch.cat(attention_scores_list, dim=0).contiguous()
    return [C]

if __name__ == '__main__':
    for index in range(repeatTime):
        batch = 4
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        head_num = 8
        head_size = 128

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeC = (head_num * sum(seqlen_squared), )

        transA = 0
        transB = 1
        m = seqlen
        k = [head_size] * batch
        n = seqlen
        lda = [hidden_size] * batch
        ldb = [hidden_size] * batch
        ldc = seqlen
        strideA = [head_size] * batch
        strideB = [head_size] * batch
        strideC = seqlen_squared

        batch = batch
        headNum = head_num

        A = torch.randn(sum_seqlen, hidden_size).npu().half()
        B = torch.randn(sum_seqlen, hidden_size).npu().half()

        shapeC = (head_num * sum(seqlen_squared), )
        ascendspeed_ops = UnpadStridedBatchMatmulOpBuilder().load()
        result = ascendspeed_ops.npu_unpad_strided_batch_matmul(A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)

        in_tensors = [A, B]
        out_tensors = [result]
        golden_out_tensors = golden_calc1(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum )

        res_compare = golden_compare(out_tensors, golden_out_tensors)
        print("res_compare:", res_compare)
