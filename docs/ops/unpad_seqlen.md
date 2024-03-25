# unpad_seqlen对外接口
功能说明：
在llama的微调场景中的unpad方案调用算子，Unpad Transformer的encoder过程中，序列长度不再按照最大长度计算，根据实际的长度进行计算（向上pad到16倍数），减少计算量。Unpad_input算子用于去掉输入数据中padding部分。

unpad方案:
在llama的微调场景中，输入序列长度（seq length）是动态的，当前的做法是对所有输入序列都按最大长度计算，存在大量冗余计算。Unpad方案是decoder过程序列长度不再按照最大长度计算，而是根据实际的长度进行计算，减少计算量。以llama-7B为例, max_seq_length=2048, num_attention_heads=32, hidden_size=4096，其整个前向过程及涉及算子如下图。其中b为batch_size，x表示当前batch的实际seq_len，X表示实际seqlen的和。

输入：
- input：必选输入，数据类型float16, shape（batch,max_seq_len,hidden_size）

输出：
- output_tensor：必选输出，数据类型float16, shape（sum(seq_len),hidden_size）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- maxSeqLen：必选输入，数据类型int32

## 案例
```python 
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
```
