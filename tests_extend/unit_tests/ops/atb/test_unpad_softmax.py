import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSoftMaxOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadSoftMax(DistributedTest):
    world_size = 1

    def supported_op_exec(self, data_input_list, seq_len, batch_size_imm):
        golden_list = [torch.softmax(data_input_list[i].reshape(-1, seq_len[i]).to(torch.float32), dim=-1).to(
            torch.float16).reshape(-1) for i in range(batch_size_imm)]
        golden = torch.cat(golden_list)
        return [golden]

    def custom_op_exec(self, data_input, seq_len, head_num_imm):
        ascendspeed_ops = UnpadSoftMaxOpBuilder().load()
        return ascendspeed_ops.npu_unpad_softmax(data_input, seq_len, head_num_imm)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_softmax(self):
        batch_size_imm = 4
        head_num_imm = 8
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
        data_input_list = [torch.randn(head_num_imm * seq_len[i] * seq_len[i]).to(
            torch.float16) for i in range(batch_size_imm)]
        data_input = torch.cat(data_input_list).contiguous()

        golden_out_tensors = self.supported_op_exec(data_input_list, seq_len, batch_size_imm)
        data_output = self.custom_op_exec(data_input.npu(), seq_len.tolist(), head_num_imm)
        data_output = data_output.cpu()
        out_tensors = [data_output]
        assert torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
