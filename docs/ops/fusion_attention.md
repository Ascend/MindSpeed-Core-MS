# fusion attention 对外接口

npu_fusion_attention(
                    query, key, value, head_num,
                    input_layout, *, pse=None,
                    padding_mask=None, atten_mask=None,
                    scale=1., keep_prob=1., pre_tokens=2147483647,
                    next_tokens=2147483647, inner_precise=0, prefix=None,
                    actual_seq_qlen=None, actual_seq_kvlen=None,
                    sparse_mode=0, gen_mask_parallel=True,
                    sync=False, pse_type=1, q_start_idx=None,
                    kv_start_idx=None)

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
- pse：可选输入，数据类型float16, bfloat16，缺省none，如果psetype为2或3的话，为float32类型。在softmax之前score的偏移量。支持 b, n, s_outer 维度广播
- padding_mask：Device侧的Tensor，暂不支持该参数
- atten_mask：Device侧的Tensor，可选参数，取值为1代表该位不参与计算（不生效），为0代表该位参与计算，数据类型支持BOOL、UINT8，数据格式支持ND格式，输入shape类型支持BNSS格式、B1SS格式、11SS格式、SS格式。varlen场景只支持SS格式，SS分别是maxSq和maxSkv
- prefix：Host侧的int array，可选参数，代表prefix稀疏计算场景每个Batch的N值。数据类型支持INT64，数据格式支持ND
- actual_seq_qlen：Host侧的int array，可选参数，varlen场景时需要传入此参数。表示query每个S的累加和长度，数据类型支持INT64，数据格式支持ND
- actual_seq_kvlen：Host侧的int array，可选参数，varlen场景时需要传入此参数。表示key/value每个S的累加和长度。数据类型支持INT64，数据格式支持ND
- sparse_mode：Host侧的int，表示sparse的模式，可选参数。数据类型支持：INT64，默认值为0，支持配置值为0、1、2、3、4、5、6、7、8。当整网的atten_mask都相同且shape小于2048*2048时，建议使用defaultMask模式，来减少内存使用量
- q_start_idx 外切时候s1方向偏移 暂未使用
- kv_start_idx 外切时候s2方向偏移 暂未使用

输出：
(Tensor, Tensor, Tensor, Tensor, int, int, int)

第1个输出为Tensor，计算公式的最终输出y，数据类型支持：FLOAT16、BFLOAT16。
第2个输出为Tensor，Softmax 计算的Max中间结果，用于反向计算，数据类型支持：FLOAT。
第3个输出为Tensor，Softmax计算的Sum中间结果，用于反向计算，数据类型支持：FLOAT。
第4个输出为Tensor，保留参数，暂未使用。
第5个输出为int，DSA生成dropoutmask中，Philox算法的seed。
第6个输出为int，DSA生成dropoutmask中，Philox算法的offset。
第7个输出为int，DSA生成dropoutmask的长度。

属性：
- scale_value：可选属性，数据类型float，缺省1。在 softmax 之前应用缩放因子。
- pse_type  可选属性 int64  范围0-3   0表示外部传入，1暂不开放， 2和3表示核内生成， 3做pse的时候会做sqrt  
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
- gen_mask_parallel：debug参数，DSA生成dropout随机数向量mask的控制开关，默认值为True：同AICORE计算并行，False：同AICORE计算串行
- sync：debug参数，DSA生成dropout随机数向量mask的控制开关，默认值为False：dropout mask异步生成，True：dropout mask同步生成

## 反向接口
输入：
- grad：必选输入，数据类型float16, bfloat16，正向attention_out的梯度输入

输出：
- grad_query：必选输出，数据类型float16, bfloat16
- grad_key：必选输出，数据类型float16, bfloat16	
- grad_value：必选输出，数据类型float16, bfloat16


## 输入限制
-输入query、key、value的B：batchsize必须相等，取值范围1~2K。
-输入query、key、value、pse的数据类型必须一致。
-输入query、key、value的input_layout必须一致。
-输入query的N和key/value的N 必须成比例关系，即Nq/Nkv必须是非0整数，Nq取值范围1~256。当Nq/Nkv > 1时，即为GQA，当Nkv=1时，即为MQA。
-输入key/value的shape必须一致。
-输入query、key、value的S：sequence length，取值范围1~512K。
-输入query、key、value的D：head dim，取值范围1~512。
-sparse_mode为1、2、3、4、5、6、7、8时，应传入对应正确的atten_mask，否则将导致计算结果错误。当atten_mask输入为None时，sparse_mode，pre_tokens，next_tokens参数不生效，固定为全计算。
-sparse_mode配置为1、2、3、5、6时，用户配置的pre_tokens、next_tokens不会生效。
-sparse_mode配置为0、4时，须保证atten_mask与pre_tokens、next_tokens的范围一致。
-prefix稀疏计算场景B不大于32，varlen场景不支持非压缩prefix，即不支持sparse_mode=5；当Sq>Skv时，prefix的N值取值范围[0, Skv]，当Sq<=Skv时，prefix的N值取值范围[Skv-Sq, Skv]。
-sparse_mode=7或者8时，不支持可选输入pse。

## flash_attention_v2 类的调用方式

```python
import os
import torch
import torch_npu
import numpy as np
import math
import mindspeed.ops.fusion_attention_v2
import npu_fusion_attention

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

# 正向接口案例
result = npu_fusion_attention(query.npu(), key.npu(), value.npu(), atten_mask.npu(), None, scale_value, q_scale,
                              head_num, input_layout, keep_prob, pre_tokens, next_tokens)

# 反向接口案例
result.backward(dx)
```
