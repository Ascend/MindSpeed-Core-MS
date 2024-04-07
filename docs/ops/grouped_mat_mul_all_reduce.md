# flash_attention对外接口

CLASS GroupedMatMulAllReduce()

```
计算逻辑：
同GroupedMatmul
```
## 前向接口：
输入：
- x：必选输入，List[Tensor]，数据类型float16，bfloat16。支持的最大长度为64个。
- weight：必选输入，List[Tensor]，数据类型float16, bfloat16。支持的最大长度为64个。
- bias：可选输入，List[Tensor]，数据类型float16, bfloat16。torch1.11版本缺省值为[], 2.1版本后缺省值为none。支持的最大长度为64个。
- group_list：可选输入，Optional[List[int64]]，缺省none。代表输入和输出M方向的matmul大小分布，支持的最大长度为64个。

输出：
- y：List[Tensor]，数据类型float16, bfloat16。支持的最大长度为64个。

属性：
- split_item：可选属性，int64。代表输入和输出是否要做tensor切分，0代表输入和输出都不用切分；1代表输入需要切分，输出不需要切分；2代表输入不需要切分，输出需要切分；3代表输入和输出都需要切分。缺省0。
- hcomm：必选属性，数据类型支持：string。表示通信域名称，Host侧标识列组的字符串。通过Hccl提供的接口获取。
- reduce_op：可选属性，数据类型支持：string。reduce操作类型。**当前版本仅支持输入"sum"。**
- comm_turn：可选属性，int64。Host侧的整型，通信数据切分数，即总数据量/单次通信量。**当前版本仅支持输入0。**


## 反向接口
无

## 输入限制
- List支持的最大长度为64； 
- 属性reduce_op仅支持输入"sum"；
- 属性comm_turn仅支持输入0；
- 通信支持2、4、8卡。

## grouped_mat_mul_all_reduce 类的调用方式
```python
    import os
    import torch
    import torch_npu
    import torch.multiprocessing as mp
    import torch.distributed as dist
    import numpy as np
    import math
	from ascendspeed.ops.grouped_mat_mul_all_reduce import ascend_grouped_mat_mul_all_reduce
    
    
    def get_hcomm_info(world_size, rank):
        dist.init_process_group(backend='hccl', rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:5001')
        print(f'device_{rank} init_process_group success.')
        if dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
        if torch.__version__ > '2.0.1':
            hcomm_info = default_pg._get_backend(torch.device('npu')).get_hccl_comm_name(rank)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank)
        print(f'device_{rank} get_hccl_comm_name success.')
        return dist, hcomm_info

    def run(rank, n_device):
        torch_npu.npu.set_device(rank)
        dist, hcomm_info = get_hcomm_info(n_device, rank)

        x1 = torch.randn(32, 256, dtype=torch.float16)
        x1_npu = x1.npu()
        w1 = torch.randn(256, 32, dtype=torch.float16)
        w1_npu = w1.npu()
        
        x2 = torch.randn(64, 128, dtype=torch.float16)
        x2_npu = x2.npu()
        w2 = torch.randn(128, 64, dtype=torch.float16)
        w2_npu = w2.npu()
        
        group_list = None
        split_item = 0
        hccl_group = hcomm_info
        reduce_op = "sum"
        comm_turn = 0

        # 正向接口案例
        x1_npu.requires_grad = False
        x2_npu.requires_grad = False
        w1_npu.requires_grad = False
        w2_npu.requires_grad = False
        y_npu = ascend_grouped_mat_mul_all_reduce([x1_npu, x2_npu],[w1_npu, w2_npu], hccl_group, bias=[], group_list=group_list, split_item=split_item, reduce_op=reduce_op, comm_turn=comm_turn)
        if rank == 0:
            for i, y in enumerate(y_npu):
                y.cpu().numpy().tofile(f"./y_{i}.bin")
    

    if __name__ == '__main__':
        n_device = 2
        mp.spawn(run, args=(n_device,), nprocs=n_device)
        
```
