import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import PadSeqLenOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUPadSeqLen(DistributedTest):
    world_size = 1

    def supported_op_exec(self, data_input, seq_len_list, max_seq_len_imm, hidden_size_imm):
        golden = torch.empty(size=[len(seq_len_list), max_seq_len_imm, hidden_size_imm], dtype=torch.float16)
        start = 0
        for i in range(len(seq_len_list)):
            golden[i][:seq_len_list[i]] = data_input[start:start + seq_len_list[i]]
            golden[i][seq_len_list[i]:] = 0
            start = start + seq_len_list[i]
        return golden

    def custom_op_exec(self, data_input, seqlen, maxseqlen):
        ascendspeed_ops = PadSeqLenOpBuilder().load()
        return ascendspeed_ops.npu_pad_seqlen(data_input, seqlen, maxseqlen)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_pad_seqlen(self):
        batch = 4
        maxseqlen = 4096
        hidden_size_imm = 4096
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        data_input = torch.randn(size=[seq_len.sum(), hidden_size_imm], dtype=torch.float16).npu().half()

        golden_out_tensors = self.supported_op_exec(data_input, seqlen, maxseqlen, hidden_size_imm)
        result = self.custom_op_exec(data_input, seqlen, maxseqlen)
        res = result.cpu()
        out_tensors = res
        assert torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
