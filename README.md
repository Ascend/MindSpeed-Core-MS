<p align="center"> <img src="sources/images/logo.png" height="90px" width="400px"></p>

# 简介

AscendSpeed 是针对华为昇腾设备的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后再通过集合通信对结果进行汇总。

昇腾提供 AscendSpeed 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

# 安装

### 1. 安装依赖

在安装**AscendSpeed**之前，请参考[版本配套表](#版本配套表)，安装最新昇腾软件栈：[https://www.hiascend.com/zh/](https://www.hiascend.com/zh/)。

| 依赖软件      |
|-----------|
| Driver    | 
| Firmware  | 
| CANN      | 
| Kernel    | 
| PyTorch   | 
| torch_npu | 
| apex      | 


### 2. 安装 AscendSpeed

如有旧版本 AscendSpeed 请先卸载，再进行安装操作。

方式一：从 git 直接安装

 ```shell
 pip install git+https://gitee.com/ascend/AscendSpeed.git
 ```

方式二：下载源码安装

 ```shell
 git clone https://gitee.com/ascend/AscendSpeed.git
 pip install -e AscendSpeed
 ```

### 3. 获取 Megatron-LM 并指定 commit id

 ```shell
 # 目前版本基于core_r0.6.0的release版本
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout core_r0.6.0
 ```

# 快速上手

以 GPT 模型为例：

方式一：若 Megatron-LM 的 commit id 与上述一致，运行[`ascendspeed`](ascendspeed/run/run.py)命令    
1. 在 Megatron-LM 目录下，运行`ascendspeed -P`命令。
    ```shell
     ascendspeed -P
    ```

2. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。
    ```shell
    bash examples/pretrain_gpt_distributed.sh
    ```

方式二：修改本地代码
1. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行`import ascendspeed.megatron_adaptor`

    ```diff
     import os
     import torch
    +import ascendspeed.megatron_adaptor
     from torch import Tensor
     from functools import partial
     from typing import Union
    ```

2. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在model_provider函数中删除`assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"`。
    ```diff
    else:
    -   assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    ```

3. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。
    ```shell
    bash examples/pretrain_gpt_distributed.sh
    ```

# 特性介绍
| 特性 | 介绍 |
| ----- | ----- |
| Megatron 数据并行 | [link](docs/features/data-parallel.md) |
| Megatron 张量并行 | [link](docs/features/tensor-parallel.md) |
| Megatron 流水并行  | [link](docs/features/pipeline-parallel.md) |
| Megatron 虚拟流水并行  | [link](docs/features/virtual-pipeline-parallel.md) |
| Megatron 序列并行  | [link](docs/features/sequence-parallel.md) |
| Megatron 重计算  | [link](docs/features/recomputation.md) |
| Megatron 分布式优化器  | [link](docs/features/distributed-optimizer.md) |
| Megatron 异步DDP  | [link](docs/features/async-ddp.md) |
| Ascend TP 重计算通信优化 | [link](docs/features/recomputation-communication.md) |
| Ascend 内存碎片优化 | [link](docs/features/memory-fragmentation.md) |
| Ascend 自适应选择重计算 | [link](docs/features/adaptive-recompute.md) |
| Ascend 计算通信并行优化 | [link](docs/features/communication-over-computation.md) |
| Ascend BF16 参数副本复用 | [link](docs/features/reuse-fp32-param.md) |
| Ascend rms_norm 融合算子 | [link](docs/features/rms_norm.md) |
| Ascend swiglu 融合算子 | [link](docs/features/swiglu.md) |
| Ascend rotary_embedding 融合算子 | [link](docs/features/rotary-embedding.md) |
| Ascend MoE token重排性能优化   | [link](docs/features/token-rearrange.md)                |
| flash attention 适配              | [link](docs/features/flash-attention.md)                 |
| 权重更新通信隐藏                        | [link](docs/features/async-ddp-param-gather.md)          |
| 【Prototype】Ulysses 长序列并行 | [link](docs/features/ulysses-context-parallel.md) |
| 【Prototype】Ring Attention 长序列并行 | [link](docs/features/ring-attention-context-parallel.md) |
| 【Prototype】Ascend MC2 | [link](docs/features/mc2.md) |
| 【Prototype】alibi | [link](docs/features/alibi.md) |
| 【Prototype】lcal_coc | [link](docs/features/communication-over-computation.md) |
| 【Prototype】其他昇腾亲和优化 | 暂无 |

# 自定义算子

| 算子                         | 介绍                                             |
|----------------------------|------------------------------------------------|
| npu_dropout_add_layer_norm | [link](docs/ops/npu_dropout_add_layer_norm.md) |
| 【Prototype】fusion_attention        | [link](docs/ops/fusion_attention.md)        |
| 【Prototype】rms_norm                   | [link](docs/ops/rms_norm.md)                   |
| 【Prototype】swiglu                     | [link](docs/ops/swiglu.md)                     |
| 【Prototype】lcal_coc                   | [link](docs/ops/lcal_coc.md)                   |
| 【Prototype】npu_mm_all_reduce_add_rms_norm | [link](docs/ops/npu_mm_all_reduce_add_rms_norm.md)                   |
| 【Prototype】npu_mm_all_reduce_add_rms_norm_ | [link](docs/ops/npu_mm_all_reduce_add_rms_norm_.md)   

# 版本配套表

**PyTorch Extension**版本号采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**PyTorch Extension**匹配的PyTorch版本，后者用于匹配CANN版本，详细匹配如下：

| AscendSpeed版本   | PyTorch版本   | Extension版本    |Python版本                               |
| ----------------- | ------------- | ------------- | --------------------------------------- |
|       1.0         |     2.1.0     |   2.1.0.post3 | Python3.8.x, Python3.9.x, Python3.10.x  |

[昇腾辅助软件](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)中有更多关于PyTorch和CANN的版本信息。

# 安全声明

[AscendSpeed 安全声明](SECURITYNOTE.md)