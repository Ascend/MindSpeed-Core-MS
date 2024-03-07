# fast_softmax对外接口
功能说明：
 在llama的微调场景中的unpad方案调用算子，针对unpad计算后的一维tensor进行softmax计算，相对于原来的只能支持相同seqlen的softmax，序列长度不再按照最大长度计算，根据实际的长度进行计算（向上pad到16倍数），更加灵活，减少计算冗余

unpad方案:
在llama的微调场景中，输入序列长度（seq length）是动态的，当前的做法是对所有输入序列都按最大长度计算，存在大量冗余计算。Unpad方案是decoder过程序列长度不再按照最大长度计算，而是根据实际的长度进行计算，减少计算量。以llama-7B为例, max_seq_length=2048, num_attention_heads=32, hidden_size=4096，其整个前向过程及涉及算子如下图。其中b为batch_size，x表示当前batch的实际seq_len，X表示实际seqlen的和。

## 前向接口：
限制条件：
- 输入矩阵：shape为（batch*headNum*sum(seq_len)）
- 输出矩阵：shape为（batch*headNum*sum(seq_len)）

输入：
- input：必选输入，数据类型float16, shape（batch*headNum*sum(seq_len)）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*sum(seq_len)）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 反向接口
输入：
- yInput：必选输入，数据类型float16, shape（batch*headNum*sum(seq_len)）
- yGrad：必选输入，数据类型float16, shape（batch*headNum*sum(seq_len)）

输出：
- output_tensor：必选输出，数据类型float16, shape（batch*headNum*sum(seq_len)）

属性：
- seqLen：必选输入，数据类型int32，vector类型，长度为batch
- headSize：必选输入，数据类型int32

## 案例
```python 正向接口案例
import torch
import torch_npu
import ascendspeed_ops

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

def test_fastsoftmax():
    batch_size_imm = 4
    head_num_imm = 8
    seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
    data_input_list = [torch.randn(head_num_imm * seq_len[i] * seq_len[i]).to(
        torch.float16) for i in range(batch_size_imm)]
    data_input = torch.cat(data_input_list).contiguous()
    data_output = ascendspeed_ops.npu_fastsoftmax(data_input.npu(), seq_len.tolist(), head_num_imm)
    # calc golden
    golden_list = [torch.softmax(data_input_list[i].reshape(-1, seq_len[i]).to(torch.float32), dim=-1).to(
        torch.float16).reshape(-1) for i in range(batch_size_imm)]
    data_output = data_output.cpu()
    golden = torch.cat(golden_list)
    out_tensors = [data_output]
    golden_out_tensors = [golden]
    success = golden_compare(out_tensors, golden_out_tensors)

if __name__ == '__main__':
    test_fastsoftmax()

```
```python 反向接口案例
import torch
import torch_npu
import ascendspeed_ops

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
    x_grad = ascendspeed_ops.npu_fastsoftmaxgrad(y_input.npu(), y_grad.npu(), seq_len.tolist(), head_num_imm)
    out_tensors = [x_grad.cpu()]
    golden_out_tensors = [golden]
    success = self.golden_compare(out_tensors, golden_out_tensors)

if __name__ == '__main__':
    test_fastsoftmaxgrad()
```
