# npu_mm_all_reduce_add_rms_norm对外接口

CLASS MatmulAllReduceAddRmsNorm()

计算逻辑：
$$
mmOut = allReduce(x1*x2 + bias)
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
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，该场景默认为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

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
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型int32
- antiquant_scale：可选输入，该场景默认为nullptr
- antiquant_offset：可选输入，该场景默认为nullptr
- dequant_scale：可选输入，数据类型int64，uint64，bfloat16
- antiquant_group_size：可选输入，该场景默认为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

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
- epsilon：可选输入，数据类型为float，缺省情况下为1e-06
- bias：可选输入，数据类型float16, bfloat16
- antiquant_scale：可选输入，数据类型float16, bfloat16
- antiquant_offset：可选输入，数据类型float16, bfloat16
- dequant_scale：可选输入，该场景默认为nullptr
- antiquant_group_size：可选输入，数据类型为int，缺省情况下为0
- comm_turn：可选输入，数据类型为int,缺省情况下为0

输出：
- y：必选输出，数据类型float16, bfloat16
- normOut：必选输出，数据类型float16, bfloat16

## 输入限制
- x2仅支持最后两轴转置情况下的非连续tensor传入，x1、residual、gamma等输入仅支持连续tensor 
- 仅支持ND数据格式
- x1支持两维或者三维，其维度为(b, s, k)或者(s, k)。
- x2仅支持两维，其维度为(k, n)，x1和x2的轴满足mm算子入参要求，k轴相等。
- bias若非空，bias为1维，其维度为(n)。
- residual仅支持三维，其维度为(b, s, n)，当x1为两维时，residual的(b*s)等于x1的s，当x1为三维时，residual的(b*s)等于x1的(b*s);residual的最后一维与x2的最后一维相等。
- gamma仅支持一维，其维度为（n），gamma的最后一维与residual的最后一维相等。
- reduce_op仅支持“sum”
- 昇腾910B AI处理器支持2、4、8卡。
- 非量化场景下，x1、x2、bias（若支持）、residual、gamma计算输入的数据类型要一致。
- 全量化场景下，若输出residual类型为FLOAT16，dequant_scale的类型为INT64、UINT64(需通过torch_npu.npu_trans_quant_param()接口对dequant_scale进行处理)；若输出residual类型为BFLOAT16，dequant_scale的类型为BFLOAT16。dequant_scale满足两种模式：
    - per_tensor模式：(1,)
    - per_channel模式：(1, n)或(n,)
- 全量化场景下，x1、x2数据类型为int8，bias（若支持）数据类型为int32,residual、gamma计算输入的数据类型要一致。
- 伪量化场景下，k、n的范围为[1, 65535]。
- 伪量化场景下，antiquant_scale满足三种模式：
    - per_tensor模式：(1,)
    - per_channel模式：(1, n)或(n,)
    - per_group模式：（ceil(k,antiquant_group_size),n）。
- antiquantOffset若非空，shape与antiquant_scale一致。
- 伪量化场景下，x2的数据类型需为int8，x1、bias（若支持）、residual、gamma、antiquant_scale、antiquant_offset计算输入的数据类型要一致。
- 伪量化场景下，antiquant_group_size取值满足取值范围[32, min(k-1, INT_MAX)]且为32倍数。

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
