import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSoftMaxOpBuilder

repeatTime=5

def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
    x.requires_grad = True
    y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
    y.retain_grad()
    w = torch.randn_like(x).to(torch.float32)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

def test_fastsoftmaxgrad():
    for i in range(repeatTime):
        batch_size_imm = 4
        head_num_imm = 8
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        golden = torch.cat(golden_list)
        
        ascendspeed_ops = UnpadSoftMaxOpBuilder().load()
        x_grad = ascendspeed_ops.npu_unpad_softmax_grad(y_input.npu(), y_grad.npu(), seq_len.tolist(), head_num_imm)
        out_tensors = [x_grad.cpu()]
        golden_out_tensors = [golden]
        
        success = golden_compare(out_tensors, golden_out_tensors)
        print("res_compare:", success)
        if not success:
            print(x_grad)
            print(golden)

if __name__ == '__main__':
    test_fastsoftmaxgrad()
