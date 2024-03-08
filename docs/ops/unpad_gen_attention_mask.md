# gen_attention_mask对外接口
功能说明：
在llama的微调场景中的unpad方案调用算子，针对unpad计算后的一维tensor进行softmax计算，相对于原来的只能支持相同seqlen的attention_mask，序列长度不再按照最大长度计算，根据实际的长度进行计算（向上pad到16倍数），更加灵活，减少计算冗余

unpad方案:
在llama的微调场景中，输入序列长度（seq length）是动态的，当前的做法是对所有输入序列都按最大长度计算，存在大量冗余计算。Unpad方案是decoder过程序列长度不再按照最大长度计算，而是根据实际的长度进行计算，减少计算量。以llama-7B为例, max_seq_length=2048, num_attention_heads=32, hidden_size=4096，其整个前向过程及涉及算子如下图。其中b为batch_size，x表示当前batch的实际seq_len，X表示实际seqlen的和。

输入：
- input：必选输入，数据类型float16, shape（batch,1,max_seq_len,max_seq_len）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*seq_len[0]*seq_len[0]+...+batch*headNum*seq_len[batch-1]*seq_len[batch-1]）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 案例
```python
import torch
import torch_npu
import ascendspeed_ops

def golden_calc(in_tensors, seqlen, headNum):
    out = []
    for i, s in enumerate(seqlen):
        for _ in range(headNum):
            out.append(in_tensors[i, :, :s, :s].flatten())
    return [torch.hstack(out)]

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

if __name__ == '__main__':
    batch = 2
    maxseqlen = 4
    seqlen = [2, 3]
    headNum = 2
    a = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
    out = []
    shapeOut = sum(map(lambda x: x**2, seqlen)) * headNum
    for _ in range(shapeOut):
        out.append(0.1)
    result = ascendspeed_ops.npu_genattentionmask(a, seqlen, headNum)
    res = result.cpu().numpy().tolist()
    out_tensors = [result]
    golden_out_tensors = golden_calc(a, seqlen, headNum)
    res_compare = golden_compare(out_tensors, golden_out_tensors)
```
