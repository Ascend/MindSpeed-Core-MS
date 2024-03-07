import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSoftMaxOpBuilder

repeatTime=5

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

def test_fastsoftmax():
    for i in range(repeatTime):
        batch_size_imm = 4
        head_num_imm = 8
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
        data_input_list = [torch.randn(head_num_imm * seq_len[i] * seq_len[i]).to(
            torch.float16) for i in range(batch_size_imm)]
        data_input = torch.cat(data_input_list).contiguous()

        ascendspeed_ops = UnpadSoftMaxOpBuilder().load()
        data_output = ascendspeed_ops.npu_unpad_softmax(data_input.npu(), seq_len.tolist(), head_num_imm)
        # calc golden
        golden_list = [torch.softmax(data_input_list[i].reshape(-1, seq_len[i]).to(torch.float32), dim=-1).to(
            torch.float16).reshape(-1) for i in range(batch_size_imm)]
        data_output = data_output.cpu()
        golden = torch.cat(golden_list)
        out_tensors = [data_output]
        golden_out_tensors = [golden]
        success = golden_compare(out_tensors, golden_out_tensors)
        print("res_compare:", success)
        if not success:
            print(data_output)
            print(golden)


if __name__ == '__main__':
    test_fastsoftmax()
