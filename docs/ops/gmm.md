# gmm对外接口

npu_gmm(x, weight, bias=None, group_list=None, group_type=0)

## 前向接口：
输入：
- x：必选输入，为tensor，数据类型float16, bfloat16, float32
- weight：必选输入，为tensor，数据类型float16, bfloat16, float32
- bias：可选输入，为tensor，数据类型float16, float32, 默认值为none。训练场景下，仅支持bias为none
- group_list：可选输入，数据类型list[int64]，默认值为none。代表分组轴（见group_type参数）上不同组的索引。
- group_type：可选输入，数据类型int64，代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组，默认值为0（限制：不支持group_type=1的场景）。

输出：
- y：必选输出，数据类型float16, bfloat16, float32


## 反向接口
输入：
- grad：必选输入，为tensor，数据类型float16, bfloat16, float32
- x：必选输入，为tensor，数据类型float16, bfloat16, float32
- weight：必选输入，为tensor，数据类型float16, bfloat16, float32
- group_list：可选输入，数据类型list[int64]，默认值为none。数据来自正向输入
  
输出：
- grad_x：必选输出，数据类型float16, bfloat16, float32
- grad_weight：必选输出，数据类型float16, bfloat16, float32
- grad_bias：当前不支持，默认为none

## gmm 类的调用方式

```python
import os
import torch
import torch_npu
import numpy as np
import math
from mindspeed.ops import gmm

num_expert, seq_len, hidden_dim = 8, 32, 256
group_list = [1, 3, 6, 10, 15, 21, 28, 32]
group_type = 0

x_shape = (seq_len, hidden_dim)
weight_shape = (num_expert, hidden_dim, seq_len)
dtype = torch.float16
x = (torch.rand(x_shape).to(dtype) - 0.5)
weight = (torch.rand(weight_shape).to(dtype) - 0.5)

# 正向接口案例
x.requires_grad = True
weight.requires_grad = True
result = gmm.npu_gmm(x.npu(), weight.npu(), bias=None, group_list=group_list, group_type=group_type)

# 反向接口案例
result.backward(torch.ones(result.shape).npu())
```
