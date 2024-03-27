import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSoftMaxOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadSoftMaxGrad(DistributedTest):
    world_size = 1

    def supported_op_exec(self, head_num, seq_len):
        x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
        x.requires_grad = True
        y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
        y.retain_grad()
        w = torch.randn_like(x).to(torch.float32)
        loss = (w * y).sum()
        loss.backward()
        return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))

    def custom_op_exec(self, y_input, y_grad, seq_len, head_num_imm):
        ascendspeed_ops = UnpadSoftMaxOpBuilder().load()
        return ascendspeed_ops.npu_unpad_softmax_grad(y_input, y_grad, seq_len, head_num_imm)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_softmax_grad(self):
        batch_size_imm = 4
        head_num_imm = 8
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = self.supported_op_exec(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        golden = torch.cat(golden_list)
        golden_out_tensors = [golden]

        x_grad = self.custom_op_exec(y_input.npu(), y_grad.npu(), seq_len.tolist(), head_num_imm)
        out_tensors = [x_grad.cpu()]
        assert torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
