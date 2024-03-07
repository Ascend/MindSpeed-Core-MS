import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadSeqLenOpBuilder

repeatTime=5

def golden_calc(data_input, seq_len_list, max_seq_len_imm, hidden_size_imm):
    golden = torch.empty(size=[sum(seq_len_list), hidden_size_imm], dtype=torch.float16)
    start = 0
    for i in range(len(seq_len_list)):
        golden[start:start + seq_len_list[i]] = data_input[i][:seq_len_list[i]]
        start = start + seq_len_list[i]
    return golden

def golden_compare(out_tensor, golden_out_tensor):
    return torch.allclose(out_tensor.float(), golden_out_tensor.float(), rtol=0.001, atol=0.001)

if __name__ == '__main__':
    for i in range(repeatTime):
        batch = 4
        maxseqlen = 2048
        hidden_size_imm = 4096
        seq_len = torch.randint(low=100, high=300, size=[batch,], dtype=torch.int32)
        seqlen = np.array(seq_len)
        data_input = torch.randn(size=[batch, maxseqlen, hidden_size_imm], dtype=torch.float16).npu().half()

        ascendspeed_ops = UnpadSeqLenOpBuilder().load()
        result = ascendspeed_ops.npu_unpad_seqlen(data_input, seqlen, maxseqlen)

        res = result.cpu()
        out_tensor = res
        golden_out_tensor = golden_calc(data_input, seqlen, maxseqlen, hidden_size_imm)
        res_compare = golden_compare(out_tensor, golden_out_tensor)
        print("res_compare:", res_compare)