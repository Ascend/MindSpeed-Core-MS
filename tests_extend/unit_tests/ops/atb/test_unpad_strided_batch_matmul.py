import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadStridedBatchMatmulOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadStridedBatchMatmul(DistributedTest):
    world_size = 1

    def supported_op_exec(self, in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
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

    def custom_op_exec(self, A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum):
        ascendspeed_ops = UnpadStridedBatchMatmulOpBuilder().load()
        return ascendspeed_ops.npu_unpad_strided_batch_matmul(A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_strided_batch_matmul(self):
        batch = 4
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        head_num = 8
        head_size = 128

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]

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

        in_tensors = [A, B]
        golden_out_tensors = self.supported_op_exec(in_tensors, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)
        
        result = self.custom_op_exec(A, B, transA, transB, m, k, n, lda, ldb, ldc, strideA, strideB, strideC, batch, headNum)
        out_tensors = [result]
        assert torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
