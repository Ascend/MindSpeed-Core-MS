
# MindSpeed MindSpore后端迁移开发指南

## 0 概述

当前大模型相关业务发展迅速，AI框架PyTorch因其编程友好受到业界大多数大模型训练、推理软件的青睐，华为昇腾也提供了基于PyTorch的[昇腾MindSpeed + 昇腾NPU训练解决方案](https://www.hiascend.com/software/mindspeed)。为此，MindSpore推出了动态图方案以及[动态图API接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html)，使用户也可以像使用PyTorch一样使用MindSpore AI框架。当前华为昇腾MindSpeed也已支持接入MindSpore AI框架作为后端引擎，打造华为全栈解决方案，使用户在友好编程的同时，也享受到华为全栈软硬结合带来的极致性能体验。

昇腾社区已经提供了[MindSpeed迁移开发](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/Mindspeedguide/mindspeed_0001.html)指导。本指南侧重提供MindSpeed MindSpore后端的迁移开发指导，帮助用户快速地将大模型训练从PyTorch后端迁移至MindSpore后端。

在介绍迁移开发前，先简要介绍MindSpore动态图和API适配工具MSAdapter，供用户了解MindSpore后端和PyTorch后端的差异，以启发用户在模型迁移开发遇到问题时进行问题排查。

### MindSpore动态图介绍

[MindSpore 动态图模式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/beginner/accelerate_with_static_graph.html?highlight=%E5%8A%A8%E6%80%81)又称PyNative模式。相比之前版本的小算子拼接方案，当前版本采用了pybind算子直调的方式，即正向算子执行直接调用底层算子接口，极大地减少了单算子执行的流程开销和数据结构转换开销，在性能上有较大提升。MindSpore动态图模式仍然是基于MindSpore的基本机制实现，因此，其与PyTorch动态图仍然存在部分机制上的差异，以下进行简要阐述。

#### 自动微分机制差异

神经网络的训练主要使用反向传播算法，自动微分是各个AI框架实现反向传播的核心机制。PyTorch使用动态计算图，在代码执行时立即运算，正反向计算图在每次前向传播时动态构建；PyTorch反向微分是命令式反向微分，符合面向对象编程的使用习惯。

MindSpore使用[函数式自动微分](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/beginner/autograd.html?highlight=%E4%BC%A0%E6%92%AD)的设计理念，提供了更接近数学语义的自动微分接口`grad`和`value_and_grad`. 与PyTorch的自动微分`Tensor.backward`机制不同，MindSpore需要针对需要自动微分的函数对象调用`grad`接口获取函数微分，并指定需要求导的输入的位置索引。`grad`和`value_and_grad`接口的使用详见 [mindspore.grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.grad.html?highlight=grad#mindspore.grad) 和 [mindspore.value_and_grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.value_and_grad.html).

#### 自定义算子

与PyTorch类似的，MindSpore动态图模式也支持了自定义算子接入，用户可以参考[基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_customopbuilder.html)了解如何进行自定义算子接入。

#### 动态图API接口差异

尽管MindSpore动态图API接口的目标是与PyTorch API保持一致，但由于框架机制等原因，部分MindSpore动态图API接口可能在参数、输入、输出、逻辑功能和特定场景等方面与PyTorch APIs存在一定差异，具体差异情况详见[PyTorch与MindSpore API映射表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html)。

### MSAdater工具介绍

[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter.git)是一款MindSpore生态适配工具，在不改变用户原有使用习惯下，将PyTorch/JAX等三方框架代码快速迁移到MindSpore生态上，帮助用户高效使用昇腾算力。该工具的基本原理是使用MindSpore动态图算子来实现PyTorch API接口，由于框架的差异性，部分接口仍存在差异或者不支持，具体支持列表详见[torch接口支持列表](https://openi.pcl.ac.cn/OpenI/MSAdapter/src/branch/master/doc/readthedocs/source_zh/docs/SupportedList.md)。

## 1 软件安装

为了便于用户理解和选择合适的MindSpeed版本，我们提供了详细的版本配套表，如表1所示。
该表详细列出了MindSpeed版本与对应的MindSpore版本及CANN版本之间的匹配关系，确保用户能够根据自身软件环境准确选择相匹配的版本，以实现最优的性能与功能支持。

<table border="0">
  <tr>
    <td> MindSpeed版本 </td>
    <td> master </td>
  </tr>
  <tr>
    <td> MindSpeed代码分支名称 </td>
    <td> core_r0.8.0：配套Megatron-LM的core_r0.8.0分支 </td>
  </tr>
  <tr>
    <td> CANN版本 </td>
    <td> CANN 8.1.RC1 </td>
  </tr>
  <tr>
    <td> MindSpore版本 </td>
    <td> 2.7.0 </td>
  </tr>
  <tr>
    <td> MSAdapter版本 </td>
    <td> master </td>
  </tr>
  <tr>
    <td> Python版本 </td>
    <td> Python3.9.x, Python3.10.x </td>
  </tr>
</table>

### 安装操作

- 安装依赖的软件

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td> 昇腾NPU驱动 </td>
    <td rowspan="5">建议下载并安装左侧软件，具体请参见《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit">CANN 软件安装指南</a>》</td>
  </tr>
  <tr>
    <td> 昇腾NPU固件 </td>
  </tr>
  <tr>
    <td> Toolkit（开发套件） </td>
  </tr>
  <tr>
    <td> Kenels（算子包） </td>
  </tr>
  <tr>
    <td> NNAL（Ascend Transformer Boost加速库） </td>
  </tr>
  <tr>
    <td> MindSpore框架 </td>
    <td> 建议下载并安装左侧软件，具体参见《<a href="https://www.mindspore.cn/install/">MindSpore 安装指南</a>》</td>
  </tr>
  <tr>
    <td> MSAdapter插件 </td>
    <td> 建议下载并安装左侧软件，具体参见《<a href="https://mindtorch.readthedocs.io/zh-cn/latest/docs/Install.html">MSAdapter 安装指南》</a></td>
  </tr>
</table>

- 下载MindSpeed-Core-MS源码master分支，执行一键适配。

  ```shell
    git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b master
    cd MindSpeed-Core-MS
    source auto_convert_xxx.sh
  ```

  **说明：** MindSpeed-Core-MS源码提供了一键适配，用户无需再手动拉取MindSpeed等仓库源码。`auto_convert_xxx.sh`中`xxx`代表使用场景，可以是`llm`（大语言模型场景）、`mm`（多模态模型场景）、`rl`（强化学习场景），具体使用见[README](https://gitee.com/ascend/MindSpeed-Core-MS)。

## 2 MindSpeed PyTorch模型迁移

对于MindSpeed PyTorch后端已支持但MindSpore后端不支持的模型，用户只需针对PyTorch后端提供的模型shell脚本进行少量适配修改，即可迁移至MindSpore后端。shell脚本适配的主体原则如下：

- 启动命修改为 `msrun`，并加入 `--ai-framework mindspore`参数；
  以下是一个启动命令适配的示例。

  适配前（PTA）：

  ```shell
    python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $MLA_ARGS \
        $ROPE_ARGS \
        $MOE_ARGS \
        $MTP_ARGS \
        $DUALPIPE_ARGS \
        $MEM_ARGS \
        --distributed-backend nccl \
        --save $CKPT_SAVE_DIR \
        --load $CKPT_LOAD_DIR \
        | tee logs/pretrain_deepseek3_671b_4k_ptd.log
  ```

  适配后（MA）：

  ```shell
    msrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $MLA_ARGS \
        $ROPE_ARGS \
        $MOE_ARGS \
        $MTP_ARGS \
        $MEM_ARGS \
        --distributed-backend nccl \
        --save $CKPT_SAVE_DIR \
        --load $CKPT_LOAD_DIR \
        --ai-framework mindspore \
        2>&1 | tee logs/ms_pretrain_deepseek3_671b_4k_ptd.log
  ```

- `DISTRIBUTED_ARGS`参数适配；
  `torchrun`命令使用的分布式参数包括 `--nproc_per_node, --nnodes, --node_rank, --master_addr, --master_port`，切换为 `msrun`后，分布式参数名需要适配调整。`msrun`使用的分布式参数和 `torchrun`的分布式参数关系见下表，各参数含义和使用详见[msrun启动](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/parallel/msrun_launcher.html?highlight=msrun)：

  | msrun 分布式参数   | 与torchrun分布式参数的关系 |
  | ------------------ | -------------------------- |
  | --local_worker_num | `=nproc_per_node`        |
  | --worker_num       | `=nproc_per_node*nnodes` |
  | --master_addr      | `=master_addr`           |
  | --master_port      | `=master_port`           |
  | --node_rank        | `=node_rank`             |
  | --log_dir          | /                          |
  | --join             | /                          |
  | --cluster_time_out | /                          |
  | --bind_core        | /                          |

  以下是一个 `DISTRIBUTED_ARGS`参数适配示例。

  适配前（PTA）：

  ```shell
  # PyTorch后端启动脚本
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost #主节点IP
  MASTER_PORT=6000
  NNODES=64
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

  DISTRIBUTED_ARGS="
      --nproc_per_node $NPUS_PER_NODE \
      --nnodes $NNODES \
      --node_rank $NODE_RANK \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT
  "
  ```

  适配后（MA）：

  ```shell
  # MindSpore后端启动脚本
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost #主节点IP
  MASTER_PORT=9110
  NNODES=64
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

  DISTRIBUTED_ARGS="
      --master_addr $MASTER_ADDR \
      --node_rank $NODE_RANK \
      --worker_num $WORLD_SIZE \
      --local_worker_num $NPUS_PER_NODE \
      --master_port $MASTER_PORT \
      --log_dir=msrun_log \
      --join=False \
      --cluster_time_out=300 \
      --bind_core=True \
  "
  ```

- 确认PyTorch shell脚本中的特性开关所对应的特性在MindSpore后端已支持。若MindSpore后端未支持，我们建议在启动脚本中关闭这些特性。

完成上述启动shell脚本适配后，用户即可尝试使用脚本拉起模型任务。

## 3 MindSpeed MindSpore进阶开发

本章节提供MindSpeed MindSpore后端的进阶开发，帮助用户在MindSpore后端下进行新模型开发或者接入自定义算子，提升开发效率。

### 新模型开发

在基于MindSpeed MindSpore后端进行新模型开发时，用户可以像使用torch API一样使用MSAdapter提供的API接口在MindSpeed中开发新模型，但请注意部分接口的差异（见[torch接口支持列表](https://openi.pcl.ac.cn/OpenI/MSAdapter/src/branch/master/doc/readthedocs/source_zh/docs/SupportedList.md)）。

若用户开发的新模型不涉及自定义加速优化特性，则基于MindSpore后端的开发与基于PyTorch后端的开发基本无差异。但若用户的新模型涉及到
自定义加速优化特性开发、特别是涉及到自定义反向微分时，则需要特别关注反向微分的定义方式。由于MindSpore采用的函数式自动微分机制，MSAdapter暂时无法支持用户以`tensor.backward()`方式完成与`tensor`有关的操作/函数的全链条微分，仍需使用MindSpore的grad接口。

### 自定义算子接入

当前MindSpeed MindSpore通过MSAdapter接入自定义算子，并通过patch机制进行函数替换使得模型训练时可以使用自定义算子。因此，接入自定义算子可以参照如下流程：

- 1. 在MSAdapter中利用MindSpore的动态图模式自定义算子机制实现自定义算子接入；
- 2. 在MindSpeed中实现patch替换。

下面以计算通信融合算子`MATMUL_ALL_REDUCE`为例介绍自定算子接入的开发流程（背景见[计算通信并行CoC特性说明](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/pytorch/features/communication-over-computation.md)）。

- 1. MSAdapter中接入自定义算子。`MATMUL_ALL_REDUCE`融合算子是硬件使能软件`CANN`提供的加速算子，需要利用MindSpore的自定义算子机制实现接入，以使Pyhon侧能够调用。

    - 首先在`MSAdapter/csrc/atb_ops`目录下新增`lcal_coc.cpp`文件，实现`matmul_all_reduce`函数并使用PyBind11将该函数注册成为Python模块接口。在开发C++算子时需要特别注意算子的输入输出。

    ```cpp
    #include <vector>
    #include "ms_extension.h"
    #include "atb_common.h"

    using LinearParallelParam = atb::infer::LinearParallelParam;

    namespace atb {
        template <>
        struct HashOpParam<LinearParallelParam> {
            void operator()(const LinearParallelParam &param) const {
                add_param_to_buf("transWeight", param.transWeight);
                add_param_to_buf("rank", param.rank);
                add_param_to_buf("rankSize", param.rankSize);
                add_param_to_buf("rankRoot", param.rankRoot);
                add_param_to_buf("hasResidual", param.hasResidual);
                add_param_to_buf("backend", param.backend);
                add_param_to_buf("commMode", param.commMode);
                add_param_to_buf("type", param.type);
                add_param_to_buf("keepIntermediate", param.keepIntermediate);
                add_param_to_buf("commDomain", param.commDomain);
            }
        };
    }  // namespace atb

    void matmul_all_reduce(const BaseTensorPtr &input1, const BaseTensorPtr &input2, const std::optional<BaseTensorPtr> &biasOpt,
                        BaseTensorPtr &output, int rank, int rankSize, const std::string &commDomain)
    {
        const BaseTensorPtr &bias = biasOpt.has_value() ? biasOpt.value() : nullptr;

        LinearParallelParam param;
        bool transB = (input1->shape()[1] != input2->shape()[0]);
        param.transWeight = transB;
        param.rank = rank;
        param.rankSize = rankSize;
        param.rankRoot = 0;
        param.hasResidual = biasOpt.has_value();
        param.backend = "lcoc";
        param.commMode = atb::infer::CommMode::COMM_MULTI_PROCESS;
        param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
        param.keepIntermediate = false;
        param.commDomain = commDomain;

        auto op = atb::OpParamCache<LinearParallelParam>::getInstance().getOperation(param, "MatMulAllReduce");
        BaseTensorPtrList inputPara = {input1, input2};
        if (biasOpt.has_value()) {
            inputPara.push_back(bias);
        }
        MsRunAtbOp(op, inputPara, {output});
    }

    PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
        m.def("matmul_all_reduce", &matmul_all_reduce, "matmul_all_reduce on ascend device", pybind11::arg("input1"),
            pybind11::arg("input2"), pybind11::arg("biasOpt"), pybind11::arg("output"), pybind11::arg("rank"), pybind11::arg("rankSize"), pybind11::arg("commDomain"));
    }
    ```

    - 然后在MSAdapter的`mindtorch/torch_npu`子模块中利用`AtbOpBuilder`(本质上是使用MindSpore `CustomOpBuilder`)对外提供`mindtorch.torch_npu.matmul_all_reduce`接口。

    ```python
    def matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain):
        return builder.load("lcal_coc").matmul_all_reduce(input1, input2, bias, output, rank, tp_size, comm_domain)
    ```

- 2. 在MindSpeed中进行patch替换（MindSpeed Torch后端已接入计算通信融合算子`matmul_all_reduce`，在MindSpore后端时对其进行替换）。
  以MindSpeed-LLM为例，在`mindspeed-llm/mindspore/mindspore_adaptor.py`的`MindSporeAdaptation`中加入patch替换逻辑：

  ```python
    from mindspeed.mindspore.ops.lcal_functional import matmul_all_reduce
    MegatronAdaptation.register('mindspeed.ops.lcal_functional.CoCOperations.matmul_all_reduce', matmul_all_reduce)
  ```

