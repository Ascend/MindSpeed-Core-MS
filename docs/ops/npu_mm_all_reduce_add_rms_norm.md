# npu_mm_all_reduce_add_rms_norm对外接口

CLASS MatmulAllReduceAddRmsNorm()

计算逻辑：
$$
mmOut = x1@x2 + bias
$$
$$
y = mmOut + residual
$$
$$
normOut = \frac{y}{RMS(y)}*gamma, RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} y_{i}^{2} + epsilon}
$$

## 非量化场景：
输入：
- x1：必选输入，数据类型float16, bfloat16	
- x2：必选输入，数据类型float16, bfloat16	
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为double，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，该场景为nullptr
- comm_turn：可选输入，数据类型为int64_t,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 全量化场景
输入：
- x1：必选输入，数据类型int8
- x2：必选输入，数据类型int8
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为double，缺省情况下为1e-06
- bias：可选输入，数据类型int32
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，数据类型int64，uint64，bfloat16
- antiquant_group_size：可选输入，该场景默认为nullptr
- comm_turn：可选输入，数据类型为int64_t,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 伪量化场景
输入：
- x1：必选输入，数据类型float16, bfloat16	
- x2：必选输入，数据类型int8
- residual：必选输入，数据类型float16, bfloat16
- gamma：必选输入，数据类型float16, bfloat16
- hcom：必选输入，数据类型string,
- reduce_op：可选输入，数据类型为string，当前仅支持sum
- epsilon：可选输入，数据类型为double，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，数据类型float16, bfloat16
- antiquant_offset：可选输入，数据类型float16, bfloat16
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，数据类型为int64_t
- comm_turn：可选输入，数据类型为int64_t,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 输入限制
- x2仅支持最后两轴转置情况下的非连续tensor传入，x1、residual、gamma等输入仅支持连续tensor 
- 仅支持ND数据格式
- x1支持两维或者三维
- x2仅支持两维
- residual仅支持三维
- gamma仅支持一维
- reduce_op仅支持“sum”

## npu_mm_all_redcue_add_rms_norm 接口的调用方式
```python
    import os
    import torch
    import torch_npu
    import numpy as np
    import math
    import torch.distributed as dist
	from ascendspeed.ops.npu_mm_all_redcue_add_rms_norm import npu_mm_all_reduce_add_rms_norm
    
    def _init_dist_hccl(rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '50000'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist
    
    b, s, k, n = 1, 16, 32, 64
    x1_shape = (b, s, k)
    x2_shape = (k, n)
    residual_shape  = (b, s, n)
    gamma_shape  = (n)
    pg = _init_dist_hccl(rank, world_size)
    group = pg.distributed_c10d._get_default_group()
    if torch.__version__ > '2.0.1':
        hcom_name = group._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
    else:
        hcom_name = group.get_hccl_comm_name(rank)
    reduce_op = "sum"
    dtype = torch.float16

    x1 = torch.rand(x1_shape).to(dtype)
    x2 = torch.rand(x2_shape).to(dtype)
    residual = torch.rand(residual_shape).to(dtype)
    gamma = torch.rand(gamma_shape).to(dtype)

    y, norm_out = npu_mm_all_reduce_add_rms_norm(x1.npu(), x2.npu(), residual.npu(), gamma.npu(),hcom_name)
```
