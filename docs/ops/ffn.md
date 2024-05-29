# ffn对外接口（只支持前向）

npu_ffn(Tensor x, Tensor weight1, Tensor weight2, str activation, *, Tensor? expert_tokens=None,
        Tensor? expert_tokens_index=None, Tensor? bias1=None, Tensor? bias2=None, Tensor? scale=None,
        Tensor? offset=None, Tensor? deq_scale1=None, Tensor? deq_scale2=None, Tensor? antiquant_scale1=None,
        Tensor? antiquant_scale2=None, Tensor? antiquant_offset1=None, Tensor? antiquant_offset2=None,
        int? inner_precise=None, ScalarType? output_dtype=None) -> Tensor

计算逻辑：
$$
y=activation(xW1+b1)W2+b2
$$

## 接口参数：
输入：
- x：必选输入，数据类型int8, float16, bfloat16
- weight1: 必选输入，数据类型int4, int8, float16, bfloat16
- weight2: 必选输入，数据类型int4, int8, float16, bfloat16
- activation: 必选输入，当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu
- expert_tokens: 可选输入，数据类型int64，不能同时输入expert_tokens和expert_tokens_index
- expert_tokens_index：可选输入，数据类型int64，不能同时输入expert_tokens和expert_tokens_index
- bias1: 可选输入，数据类型int32, float16, float32
- bias2: 可选输入，数据类型int32, float16, float32
- scale: 可选输入，数据类型float32
- offset: 可选输入，数据类型float32
- deq_scale1：可选输入，数据类型uint64, bfloat16
- deq_scale2：可选输入，数据类型uint64, bfloat16
- antiquant_scale1: 可选输入，数据类型float16, bfloat16
- antiquant_scale2: 可选输入，数据类型float16, bfloat16
- antiquant_offset1: 可选输入，数据类型float16, bfloat16
- antiquant_offset2: 可选输入，数据类型float16, bfloat16
- inner_precise：可选输入
- output_dtype：可选输入

输出：
- y：必选输出，数据类型float16, bfloat16

## npu_ffn 接口的调用方式
```python
import os
import torch
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from mindspeed.ops import ffn

os.environ["ENABLE_ACLNN"] = "true"
config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

ACTIVATION_TYPE = 'relu'

EXPERT_TOKENS_MODE_NONE = 0
EXPERT_TOKENS_MODE_TOKENS = 1
EXPERT_TOKENS_MODE_TOKENS_INDEX = 2


class FFNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, weight1, weight2, activation, expert_tokens=None, expert_tokens_index=None,
                bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None,
                antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None,
                inner_precise=0):
        return ffn.npu_ffn(x, weight1, weight2, activation,
            expert_tokens=expert_tokens, expert_tokens_index=expert_tokens_index,
            bias1=bias1, bias2=bias2, inner_precise=inner_precise)


def test_ffn(tokens_mode):
    M = 512
    K1 = 256
    N1 = 1024
    K2 = N1
    N2 = K1

    dtype = torch.float16
    bias_dtype = torch.float16 if dtype == torch.float16 else torch.float32

    expert_tokens = None
    expert_tokens_index = None

    if tokens_mode == EXPERT_TOKENS_MODE_NONE:
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
    elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS:
        E = 8
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
        expert_tokens = [64, 64, 64, 64, 64, 64, 64, 64]
        expert_tokens = torch.tensor(expert_tokens, dtype=torch.int64)
    elif tokens_mode == EXPERT_TOKENS_MODE_TOKENS_INDEX:
        E = 8
        x = torch.empty(M, K1, dtype=dtype).uniform_(-1.0, 1.0)
        weight1 = torch.empty(E, K1, N1, dtype=dtype).uniform_(-0.1, 0.1)
        weight2 = torch.empty(E, K2, N2, dtype=dtype).uniform_(-0.1, 0.1)
        bias1 = torch.empty(E, N1, dtype=bias_dtype).uniform_(-0.1, 0.1)
        bias2 = torch.empty(E, N2, dtype=bias_dtype).uniform_(-0.1, 0.1)
        expert_tokens_index = [64, 128, 192, 256, 320, 384, 448, 512]
        expert_tokens_index = torch.tensor(expert_tokens_index, dtype=torch.int64)

    x = x.npu()
    weight1 = weight1.npu()
    weight2 = weight2.npu()
    bias1 = bias1.npu()
    bias2 = bias2.npu()

    if expert_tokens != None:
        expert_tokens = expert_tokens.npu()
    if expert_tokens_index != None:
        expert_tokens_index = expert_tokens_index.npu()

    model = FFNModel().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    y = model(x, weight1, weight2, ACTIVATION_TYPE, expert_tokens=expert_tokens,
        expert_tokens_index=expert_tokens_index, bias1=bias1, bias2=bias2)
    print('y.shape:', y.shape)


if __name__ == '__main__':
    test_ffn(EXPERT_TOKENS_MODE_NONE)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS)
    test_ffn(EXPERT_TOKENS_MODE_TOKENS_INDEX)
```