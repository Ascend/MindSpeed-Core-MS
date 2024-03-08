<p align="center"> <img src="sources/images/logo.png" height="90px" width="400px"></p>

# 简介

AscendSpeed 是针对华为昇腾设备的大模型加速库。

大模型训练是一种非常复杂的过程，涉及到许多技术和挑战，其中大模型训练需要大量的显存资源是一个难题，对计算卡提出了不小的挑战。
为了在单个计算卡显存资源不足时，可以通过多张计算卡进行计算，业界出现了类似 Megatron、DeepSpeed 等第三方大模型加速库，对模型、输入数据等进行切分并分配到不同的计算卡上，最后在通过集合通信对结果进行汇总。

昇腾提供 AscendSpeed 加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。

# 安装

### 1. 安装依赖

| 依赖软件      | [验证版本](https://www.hiascend.com/zh/) |
|-----------|--------------------------------------|
| Python    | 3.8                                  |
| driver    | Ascend HDK 23.0.0                    |
| firmware  | Ascend HDK 23.0.0                    |
| CANN      | CANN 7.0.0                           |
| kernel    | CANN 7.0.0                           |
| PyTorch   | 2.1.0                                |
| torch_npu | release v5.0.0                       |
| apex      | v5.0.rc3                             |

### 2. 安装 AscendSpeed

方式一：从 git 直接安装

 ```shell
 pip install git+https://gitee.com/ascend/AscendSpeed.git
 ```

方式二：下载源码安装

 ```shell
 git clone https://gitee.com/ascend/AscendSpeed.git
 cd AscendSpeed
 pip install -e .
 ```

### 3. 获取 Megatron-LM 并指定 commit id

 ```shell
 git clone https://github.com/NVIDIA/Megatron-LM.git
 cd Megatron-LM
 git checkout bcce6f54e075e3c3374ea67adefe54f3f2da2b07
 ```

# 快速上手

### Megatron-LM NPU 适配

AscendSpeed 对 Megatron-LM
的基本功能进行了适配，已适配如下特性（特性使用请查看[官方文档](https://github.com/NVIDIA/Megatron-LM)）：

- 张量并行
- 流水线并行
- 序列并行
- 重计算
- 分布式优化器
- 异步分布式数据并行

使用方式：

1. 在 Megatron-LM 目录下修改`pretrain_gpt.py`文件，在`import torch`下新增一行`import ascendspeed.megatron_adaptor`

    ```diff
     import os
     import torch
    +import ascendspeed.megatron_adaptor
     from torch import Tensor
     from functools import partial
     from typing import Union
    ```

2. 在 Megatron-LM 目录下，准备好训练数据，并在示例脚本中填写对应路径，然后执行。
    ```shell
    bash examples/pretrain_gpt_distributed.sh
    ```

# 特性介绍

### TP重计算通信优化

优化重计算中通信算子，提升模型性能。  
具体信息请查看：[link](docs/features/recomputation-communication.md)

### 内存碎片优化

通过对不同生命周期的 tensor 进行分别管理，以减少显存碎片。  
具体信息请查看：[link](docs/features/memory-fragmentation.md)

### 自适应选择重计算

自动调整训练显存大小，选择重计算策略，提高模型训练的性能。  
具体信息请查看：[link](docs/features/adaptive-recompute.md)

### ATB算子

| 算子                         | 介绍                                             |
|----------------------------|------------------------------------------------|
| flash_attention            | [link](docs/ops/flash_attention.md)            |
| npu_dropout_add_layer_norm | [link](docs/ops/npu_dropout_add_layer_norm.md) |
| pad_seqlen                 | [link](docs/ops/pad_seqlen.md)                 |
| rms_norm                   | [link](docs/ops/rms_norm.md)                   |
| swiglu                     | [link](docs/ops/swiglu.md)                     |
| unpad_gen_attention_mask   | [link](docs/ops/unpad_gen_attention_mask.md)   |
| unpad_rope                 | [link](docs/ops/unpad_rope.md)                 |
| unpad_seqlen               | [link](docs/ops/unpad_seqlen.md)               |
| unpad_softmax              | [link](docs/ops/unpad_softmax.md)              |
| unpad_strided_batch_matmul | [link](docs/ops/unpad_strided_batch_matmul.md) |

# 安全声明

## 系统安全加固

用户可在运行系统配置时开启 ASLR（级别2）以提高系统安全性，保护系统随机化开启。  
可参考以下方式进行配置：

```
echo 2 > /proc/sys/kernel/randomize_va_space
```

## 运行用户建议

基于安全性考虑，建议您在执行任何命令时，都尽量使用非 root 账户执行，遵循权限最小化原则。

## 文件权限控制

- 建议用户对训练所需文件、训练过程中保存的文件、用户个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，例如多用户共享数据集场景下的数据集文件写权限控制、profiler 等场景产生数据文件权限控制等，设定的权限建议参考[表1 文件权限参考](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0168.html#ZH-CN_TOPIC_0000001768428486__zh-cn_topic_0000001808103017_table12631223865) 进行设置。
- 原生 Megatron-LM 以及 PyTorch 框架运行中所生成的文件权限依赖系统设定，如 Megatron-LM 生成的数据集索引文件、torch.save 接口保存的文件等。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考[表1 文件权限参考](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0168.html#ZH-CN_TOPIC_0000001768428486__zh-cn_topic_0000001808103017_table12631223865)。可使用 umask 控制默认权限，避免提权等安全风险，建议用户将主机（包括宿主机）和容器中的 umask 设置为 027 及其以上，提高安全性。
- torch_npu 中 profiler 工具会生成性能记录文件，生成的文件权限为 640 ，文件夹权限为 750 ，用户可根据需要自行对生成后的相关文件进行权限控制。
- 运行时 CANN 可能会缓存算子编译文件，存储在运行目录下的`kernel_meta_*`文件夹内，加快后续训练的运行速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[表1 文件权限参考](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0168.html#ZH-CN_TOPIC_0000001768428486__zh-cn_topic_0000001808103017_table12631223865) 进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`， 注意对`<FILE>`文件及目录做好权限管控。

建议用户根据自身需要，参考[表1 文件权限参考](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0168.html#ZH-CN_TOPIC_0000001768428486__zh-cn_topic_0000001808103017_table12631223865) 对各类文件进行加固。

## 数据安全声明

- PyTorch 和 torch_npu 提供的 profiler 性能分析工具和 torch_npu 提供 AOE 性能调优工具都会在本地生成性能拆解数据。 用户需加强对相关数据的保护，需要模型性能调优时使用，调优完成后及时关闭，AOE 和 Profiler 工具具体细节请参考《[PyTorch 模型迁移和训练指南](https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/AImpug_0001.html)》。
- PyTorch 使用过程中需要加载和保存数据，部分接口使用风险模块 pickle，可能存在数据风险，如 torch.load、torch.distributed.scatter_object_list 等接口，可参考 [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load)、[collective-functions](https://pytorch.org/docs/stable/distributed.html#collective-functions) 了解具体风险。
- AscendPyTorch 依赖 CANN 的基础能力实现 AOE 性能调优、算子 dump、日志记录等功能，用户需要关注上述功能生成文件的权限控制。

## 编译安全声明

AscendSpeed 中各类融合算子通过调用 PyTorch 中的 cpp_extension 特性进行编译，编译结果会默认缓存到 `~/.cache/torch_extensions` 目录下，建议用户根据自身需要，参考[表1 文件权限参考](https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0168.html#ZH-CN_TOPIC_0000001768428486__zh-cn_topic_0000001808103017_table12631223865) 对生成文件做好权限控制。

## 运行安全声明

- 建议根据运行环境资源状况编写训练脚本。若训练脚本与资源状况不匹配，如数据集的 size 超出内存容量/ NPU 存储容量等，可能引发错误并导致进程退出。
- AscendSpeed、PyTorch 以及 torch_npu 在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看 CANN 日志、解析生成的 Core Dump 文件等方式。

## 通信安全声明

作为计算集群的完全控制者，请务必注意集群节点间的通信安全，比如做好组网设计并采取相关安全措施。建议在内部网络下部署计算集群，从而避免公网环境下的诸多安全风险。

## 端口安全声明

AscendSpeed 不主动开放端口，对于原生 PyTorch 开放的相关端口，您可以参考其官方文档进行设置。在单机训练的情况下，不建议开放全局端口。具体的通信矩阵可以参考[附录B 通信矩阵](#B-通信矩阵)。

## 附录

### 表1 文件权限参考

| 类型           | linux权限参考值      | 备注                        |
|--------------|-----------------|---------------------------|
| 文件夹 / 目录     | 750 (rwxr-x---) | 对于共享目录可为755               |
| 数据集文件        | 640 (rw-r-----) | 对于共享数据集文件可为644            |
| checkpoint文件 | 640 (rw-r-----) | 对于checkplint等生成文件可以设置为640 |
| 程序文件         | 440 (r--r-----) | 除非开发调试场景，正常运行时程序文件不应再次修改  |
| 可执行脚本        | 750 (rwxr-x---) | 针对可执行脚本设置750              |

### 表2 通信矩阵

|        源设备         |  源IP   |                                   源端口                                    |        目的设备        |  目的IP  |                                                          目的端口（侦听）                                                           | 协议  |                                         端口说明                                          |                                                                                                                                备注                                                                                                                                |
|:------------------:|:------:|:------------------------------------------------------------------------:|:------------------:|:------:|:---------------------------------------------------------------------------------------------------------------------------:|:---:|:-------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP |        当用户不使用**测试示例脚本**，则默认29500/29400。用户可调用`torch.distributed.launch`函数，通过传入的`--master_port`自由指定1024-65535之间未被占用的端口        | TCP | 源端口与目的端口均用于收发数据。对于静态分布式场景（backend=static）默认端口为29400；对于动态分布式场景（backend=c10d）中默认端口29500 |                                                                       megatron_npu本身不开启端口，该通信过程由开源软件Pytorch控制，配置方式可参考其官方文档：https://pytorch.org/docs/stable/distributed.html#launch-utility                                                                       |
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP | 当使用`pretrain_gpt_distributed*`系列测试示例脚本，脚本对`torch.distributed.launch`传入的`--master_port`为**6000**，用户可以自由指定1024-65535之间未被占用的端口 | TCP |             原生Pytorch（调用`torchrun`、`torch.distributed.launch`）通信需要，用于收发数据             |                                                                                                  和第一条记录所述为同一端口，这里特别说明**测试示例脚本**对Pytorch开启的master_port默认配置为6000                                                                                                   |
| 运行torch_npu进程的计算设备 | 设备地址IP | 操作系统自动分配，分配范围由操作系统决定，如ubuntu是采用`/proc/sys/net/ipv4_local_port_range`文件指定 | 运行torch_npu进程的计算设备 | 设备地址IP |  当使用`test_gpt_distributed*`系列测试示例脚本，脚本对`torch.distributed.launch`传入的`--master_port`为**60035**，用户可以自由指定1024-65535之间未被占用的端口   | TCP |             原生Pytorch（调用`torchrun`、`torch.distributed.launch`）通信需要，用于收发数据             |                                                                                                  和第一条记录所述为同一端口，这里特别说明**测试示例脚本**对Pytorch开启的master_port默认配置为60035                                                                                                  |
| 运行torch_npu进程的计算设备 | 设备地址IP |                             请参见备注中的CANN官方文档                              | 运行torch_npu进程的计算设备 | 设备地址IP |                                                       请参见备注中的CANN官方文档                                                       | TCP |                                    请参见备注中的CANN官方文档                                    | 该通信过程完全由HCCL组件控制，端口范围可参考文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha001/ref/envref/envref_07_0065.html CANN通信文档：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha001/ref/hcclapiref/hcclapi_07_0001.html |

