import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSeqLenOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadSeqLen(DistributedTest):
    world_size = 1

    def supported_op_exec(self, data_input, seq_len_list, hidden_size_imm):
        golden = torch.empty(size=[sum(seq_len_list), hidden_size_imm], dtype=torch.float16)
        start = 0
        for i in range(len(seq_len_list)):
            golden[start:start + seq_len_list[i]] = data_input[i][:seq_len_list[i]]
            start = start + seq_len_list[i]
        return golden

    def custom_op_exec(self, data_input, seqlen, maxseqlen):
        ascendspeed_ops = UnpadSeqLenOpBuilder().load()
        return ascendspeed_ops.npu_unpad_seqlen(data_input, seqlen, maxseqlen)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_seqlen(self):
        batch = 4
        maxseqlen = 2048
        hidden_size_imm = 4096
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        data_input = torch.randn(size=[batch, maxseqlen, hidden_size_imm], dtype=torch.float16).npu().half()

        golden_out_tensor = self.supported_op_exec(data_input, seqlen, hidden_size_imm)
        result = self.custom_op_exec(data_input, seqlen, maxseqlen)
        out_tensor = result.cpu()
        assert torch.allclose(out_tensor.float(), golden_out_tensor.float(), rtol=0.001, atol=0.001)
