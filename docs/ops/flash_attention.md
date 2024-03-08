# flash_attention对外接口

CLASS FlashAttention()

```
计算逻辑：
同Attention
```
## 前向接口：
输入：
- query：必选输入，数据类型float16, bfloat16	
- key：必选输入，数据类型float16, bfloat16	
- value：必选输入，数据类型float16, bfloat16
- atten_mask：可选输入，数据类型bool，缺省none。在softmax之前drop的mask
- alibi_mask：可选输入，数据类型float16, bfloat16，缺省none。在softmax之前score的偏移量。支持 b, n, s_outer 维度广播

输出：
- attention_out：必选输出，数据类型float16, bfloat16

属性：
- scale_value：可选属性，数据类型float，缺省1。在 softmax 之前应用缩放因子。
- q_scale：可选属性，数据类型float，缺省1。query的缩放因子。
- head_num：可选属性，数据类型int64，缺省1。输入 shape 中的 n。
- io_layout：可选属性，数据类型string	缺省“BNSD”。可支持“BSH”, “SBH”, “BNSD”
   - h = n * d
   - BNSD 下输入shape：query（b, n, s, d）   key（b, n, s, d） value（b, n, s, d） atten_mask (s, s,) alibi_mask（b(1), n(1), s(1), s）
   - BSH 下输入shape：query（b, s, h）   key（b, s, h） value（b, s, h） atten_mask (s, s,) alibi_mask（b(1), n(1), s(1), s）
   - SBH 下输入shape：query（s, b, h）   key（s, b, h） value（s, b, h） atten_mask (s, s,) alibi_mask（b(1), n(1), s(1), s）
- keep_prob：可选属性，数据类型float，默认值为1.0。在 softmax 后的保留比例。
- pre_tokens：可选属性，数据类型int64，默认值为2147483647。atten_mask 输入的左边第一列 False 的数量。
- next_tokens：可选属性，数据类型int64，默认值为1。atten_mask 输入的上边第一行 False 的数量。
- precise_mode：可选属性，数据类型int64，缺省0。0内存优化，1性能优化



## 反向接口
输入：
- grad：必选输入，数据类型float16, bfloat16，正向attention_out的梯度输入

输出：
- grad_query：必选输出，数据类型float16, bfloat16
- grad_key：必选输出，数据类型float16, bfloat16	
- grad_value：必选输出，数据类型float16, bfloat16


## 输入限制
- d 必须是 <=128 并且为16的倍数； 
- s 必须是 256 的倍数；
- io_layout 为 BSH 时：n * d <= 524280
- io_layout 为 SBH 时：b * n * d <= 524280
- query, key, value, alibi_mask, attention_out, grad, grad_query, grad_key, grad_value 数据类型保持一致
- q_scale 属性暂不支持
- precise_mode 暂时只支持 0:内存优化

## flash_attention 类的调用方式
```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import math
	import ascendspeed.ops
    
    batch_size, head_num, seq_size, head_dim = 1, 32, 2048, 128
    input_layout = "BNSD"
    keep_prob = 1.0
    pre_tokens = 65536
    next_tokens = 1
    scale_value = 0.088
    q_scale = 1.0

    input_shape = (batch_size, head_num, seq_size, head_dim)
    softmax_shape = (batch_size, head_num, seq_size)
    mask_shape = (1, 1, seq_size, seq_size)
    dtype = torch.float16
    attention_score_grad = (torch.rand(input_shape).to(dtype) - 0.5)
    query = (torch.rand(input_shape).to(dtype) - 0.5) * 5
    key = (torch.rand(input_shape).to(dtype) - 0.5) * 5
    value = (torch.rand(input_shape).to(dtype) - 0.5) * 5
    atten_mask = np.tri(seq_size, k=-next_tokens) + np.tri(seq_size, k=-pre_tokens).transpose()
    alibi_mask = torch.zeros((batch_size, head_num, seq_size, seq_size)).to(dtype)
    atten_mask = torch.from_numpy(atten_mask).to(torch.bool)

    // 正向接口案例
	query.requires_grad = True
	key.requires_grad = True
	value.requires_grad = True
    result = ascendspeed.ops.flash_attention(query.npu(), key.npu(), value.npu(), atten_mask.npu(), None, scale_value, q_scale, head_num, input_layout, keep_prob, pre_tokens, next_tokens)
    

    // 反向接口案例
	result.backward(torch.ones(result.shape).npu())
```
