# 大模型动态图训练性能调优指南

大模型具有更多的参数量和更复杂的模型结构，在训练过程中需要耗费更多的计算资源和时间。尤其在AI框架动态图模式下易于开发调试带来了极大的灵活性，但也暴露了更多的性能及内存问题。本文将重点介绍昇思动态图下，大模型性能及内存常见的调优方法，以及如何使用工具快速分析并解决瓶颈，帮助大模型开发者们快速上手模型性能调优，提升训练效率，降低训练成本。

## 典型性能统计指标

1. 单步时间(s)：执行完一个完整BS的时间。
2. 吞吐(Samples/s)：网络模型在单位时间内可以处理的最大输入的训练样本数据。$throughput = BS * N / step\_time$。其中，BS为每个数据并行维度的batch size大小，N为集群中数据并行维度的大小，step_time为在分布式集群中，执行完一个完整BS的时间（单位为s）。
3. MFU(%)：Model FLOPs Utilization，即模型算力利用率，是指模型一次前反向计算消耗的矩阵算力与机器算力的比值。它直接反映了模型在训练过程中对计算资源的有效利用程度。MFU的值受到多种因素的影响，包括但不限于：
    - 模型架构：不同架构的模型在分布式训练中的算力利用率可能存在显著差异。例如，Transformer架构的模型由于其并行性和矩阵运算密集的特点，通常能够在分布式训练中实现较高的MFU。
    - 分布式训练策略：包括数据并行、模型并行、流水线并行等不同的并行策略及这些策略的具体实现方式（如梯度累积、张量并行等），都会对MFU产生重要影响。
    - 硬件环境：显卡型号、数量、网络连接速度等硬件因素都会限制分布式训练的性能和算力利用率。
    - 软件优化：包括编译器优化、库函数优化、自动混合精度训练等软件层面的优化措施，也能在一定程度上提升MFU。
    - 数据集和批次大小：数据集的大小和复杂性，以及训练时使用的批次大小，也均会影响每次迭代中的计算量和算力利用率。
4. 线性度：单卡训练扩展到多卡、多机多集群后的效率度量指标称为线性度，又名加速比。一般根据吞吐率计算得到。即集群线性度为多机总吞吐率/（单机吞吐率*集群卡数）。线性度的取值范围为0~1，数值越接近于1，其性能指标越好，一般大于0.8认为较优。

## MindSpore动态图机制介绍

### 动态图并行库

在大模型训练中，由于数据量和模型复杂度的增加，单个计算节点的计算能力难以满足训练的需求。为了提高训练效率和加速训练过程，通常采用并行策略来将计算任务分配给多个计算节点进行计算。MindSpeed-Core-MS提供了动态图下常见的并行策略如DP(数据并行)、MP(模型并行)和PP(流水线并行)等通用型并行策略，同时提供了对标Megatron的细粒度配置方式，助力用户在昇腾设备上高效实现大模型训练。

- DP（Data Parallelism）将训练数据划分为多个小批次，并将这些小批次分配给不同的计算设备进行并行处理。每个设备都有一份模型参数的副本，并根据自己所处理的数据进行参数更新。最后，通过对各个设备的参数进行平均或累积，得到最终的模型参数。
- MP（Model Parallelism）将模型分解为多个子模型，并将这些子模型分配给不同的计算设备进行并行处理。每个设备只负责处理子模型的一部分计算，然后将结果传递给其他设备进行进一步的计算。最后，通过协同合作，得到最终的模型输出。
- PP（Pipeline Parallelism）将模型的不同层或模块分配给不同的计算设备进行并行处理。每个设备只负责处理模型的一部分计算，并将结果传递给下一个设备进行进一步的计算。通过这种方式，可以将计算任务划分为多个阶段，提高整体的计算效率。

具体并行策略的配置方法请参考：[动态图并行-GPT模型预训练开发指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/source_zh_cn/usage/pretrain_gpt.md)

### 动态图PyNative框架

MindSpore PyNative动态图模式采用pybind算子直调方法，提升API性能。2.3版本前，大部分API使用了小算子进行拼接，同时采用了单算子子图进行执行，需要进行单算子子图构图、编译优化、算子选择、算子编译等一系列操作，首轮性能较差。针对这两点进行性能优化，提出了算子直调的方式，即正向算子执行直接pybind调用到底层算子接口，减少整体流程和数据结构转换开销。不同API，性能提升0.5~4倍。此外，提供了基础分布式能力接口，如硬件相关接口、重计算接口、通信基础接口等。
详细介绍见：[动态图支持算子直调](https://www.mindspore.cn/news/newschildren?id=3223&type=versionRelease)

### 典型模型性能数据及推荐配置

- 语言模型盘古38BV3预训练
    - 模型参数量：38B
    - 集群配置：16机128卡，Ascend
    - 推荐配置：dp4 tp8 pp4 vpp4，GBS256 MBS2 micro_num=32，关闭重计算
    - 性能数据：单步耗时11.5s/step，MFU 45%+

## 性能调优

为了优化前文介绍的性能度量指标，提升训练效率，一般从如下几个维度拆解优化。

- 数据处理耗时：指模型加载训练数据和权重的时间，包括将数据从硬件存储设备读取到CPU、在CPU中进行数据的预处理、以及CPU数据传输到NPU的过程。对于需要切分到若干张NPU上的模型，数据加载时间还包括从一张NPU广播到其他NPU上的时间。
- Host下发耗时：指Python侧脚本逻辑及api接口launch的时间，一般希望这部分耗时与Device执行流水起来，避免Device空闲。
- Device执行耗时：NPU侧执行各个算子计算逻辑的时间。
- 通信耗时：指在分布式训练中，设备之间进行通信传输的时间，可以通过优化网络拓扑、减少通信频率等方式来减少通信耗时。同时，通信和计算通常可以并行执行，并行库提供的并行技术会保证部分通信时间被掩盖。

### 用户编程checklist

- 关闭确定性计算

    ```python
    ms.set_context(deterministic='OFF')
    ```

- 关闭流同步

    ```python
    ms.set_context(pynative_synchronize=False)
    ```

- 使用高性能API
    - mint：MindSpore提供了对标PyTorch的mint系列接口，绝大多数情况下，其性能会持平或高于原ops系列接口。详情参考API列表：[mint接口列表](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore.mint.html)
    - 大模型融合算子，如RotaryPositionEmbedding、Swiglu等。
- 避免冗余的数据拷贝
    - 尽量采用原地更新接口，减少冗余Tensormove操作
    - 减少不必要的转连续操作：当前aclnn算子大多支持非连续输入，尽量减少在脚本中大量使用.contiguous()，或可以先is_contiguous()判断后再调用。
    - 避免频繁数据拷贝：需要数据拷贝时，尽量采用.from_numpy接口，当数据连续时，会通过免拷贝方式将Numpy数组转换为张量。[mindspore.Tensor.from_numpy](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/Tensor/mindspore.Tensor.from_numpy.html)
- 减少原生Python累加等函数使用，如.sum()（建议替换成mint.sum()）

### 性能配置建议

- Batch Size：
    - Global-batch-size：在固定其余参数不变的情况下，增大GBS会降低空泡占比从而提高MFU。
    - Micro-batch-size：对于已给定GBS后，增加MBS会减少流水线中的微批次数量，导致流水线气泡增大；然而，增加微批大小也可以通过增加执行内核的算术强度来提高NPU利用率。这两个因素相互矛盾，需要实验决策，一般在显存足够的情况下，micro-batch-size=2大概率较优。

    ```yaml
    # Global-batch-size = DP * batch_size * micro_batch_num
    dataset_config.batch_size   # 训练和评估时的批次大小，默认值：1
    dataset_config.micro_batch_num   # 微批次大小，默认值：1
    ```

- VPP：在流水线并行场景，由于切分粒度较大，导致空泡占比较大，因此可以在设备数量不变的情况下，分出更多的流水线阶段，即虚拟pipeline，以更多的通信量，换取空泡比率降低。

    ```yaml
    parallel_config.virtual_pipeline_model_parallel_size    # 虚拟流水线并行（VPP）的大小，默认值：None
    parallel_config.num_layers_per_virtual_pipeline_stage    # 每个VPP的层数，无缺省
    ```

- 重计算：在显存足够的情况下，可以完全关闭重计算。否则可开启部分重计算。

    ```yaml
    parallel_config.recompute   #（按层）完全重计算，用于降低训练时内存占用，默认值：None
    parallel_config.select_recompute   #（按算子）选择重计算，默认值：None
    parallel_config.select_comm_recompute   #选择通信重计算，默认值：None
    ```

具体配置方式见：[动态图并行-GPT模型预训练开发指南](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/source_zh_cn/usage/pretrain_gpt.md)

### 高阶优化特性

- 负载均衡：MindSpore提供了SAPP（Symbolic Automatic Parallel Planner）自动负载均衡工具。输入模型的内存和时间信息，以及部分流水线并行性能相关的超参（如重计算对性能的影响），工具将自行构建线性规划问题，通过全局求解的方式，为大模型自动生成流水线并行中的stage-layer配比，调整各layer重计算策略，自动优化集群算力和内存利用率，降低空等时间，实现Pipeline并行分钟级策略寻优，大幅度降低性能调优成本，提升端到端训练性能。详见：[SAPP流水线负载均衡](https://gitee.com/ascend/MindSpeed-Core-MS/blob/r0.1.0/docs/source_zh_cn/usage/pipeline_balance.md)。
- 非饱和切分：用于减少跨机通信，提升多机场景线性度及性能。在initialize_model_parallel中配置zero_shard_size方式使能，其中zero_shard_size∈(1, dp_size]且DP必须可以被zero_shard_size整除。

    ```python
    initialize_model_parallel(tensor_model_parallel_size=parallel_config.tensor_model_parallel_size, zero_shard_size=zero_shard_size)
    ```

- JIT：将Python函数编译为可调用的MindSpore静态图，MindSpore可以在运行时对图进行优化，提升该模块性能，一般选择在优化器构造函数处添加@jit标签。详见：[jit介绍](https://www.mindspore.cn/docs/zh-CN/r2.4.0/api_python/mindspore/mindspore.jit.html)

### 性能分析案例

[MindStudio Insight用户指南](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/msinsightug/msascendinsightug/Insight_userguide_0002.html)

[MindStudio Insight性能调试样例 (Ascend)](https://www.mindspore.cn/docs/zh-CN/r2.4.0/model_train/optimize/profiler.html)

## 内存优化

训练过程中内存开销一般可分为模型参数、优化器状态、前向计算结果和临时变量存储几个维度。内存包括Host侧内存及Device侧显存，因为显存资源较为稀缺，一般开发者更多关注在显存优化。

### 用户编程checklist

- 自定义算子反向数据按需保留：自定义算子易引入内存生命周期过长问题。用户实现自定义算子时，可调用used_bprop_inputs仅保存反向需要用到的数据。如下代码所示，其反向仅需要dout，即index=2，则需要在__init__函数中添加标识，此时在动态图正向执行完仅会保留dout数据。

    ```python
    class ReduceScatterToSequenceParallelRegion(nn.Cell):
        "Reduce scatter the input from the model parallel region."

        def __init__(self, need_to_swapaxes):
            super(ReduceScatterToSequenceParallelRegion, self).__init__()
            self.world_size = get_tensor_model_parallel_world_size()
            self.need_to_swapaxes = need_to_swapaxes
            if self.world_size > 1:
                self.tp_group = get_tensor_model_parallel_group()
            self.used_bprop_inputs = [2]   #bprop仅需要保留dout，则仅将dout的index传入标记

        def construct(self, input_):
            if self.world_size == 1:
                return ops.stop_gradient(input_)
            if self.need_to_swapaxes:
                input_ = input_.swapaxes(0, 1)
            output = comm_func.reduce_scatter_tensor(input_, group=self.tp_group)[0]
            if self.need_to_swapaxes:
                output = output.swapaxes(0, 1)
            return output

        # pylint: disable=W0613, C0111
        def bprop(self, *args):
            dout = args[-1]
            if self.world_size == 1:
                return dout
            if self.need_to_swapaxes:
                dout = dout.swapaxes(0, 1)
            output = comm_func.all_gather_into_tensor(dout, group=self.tp_group)[0]
            if self.need_to_swapaxes:
                output = output.swapaxes(0, 1)

            return (output,)
    ```

- 打开虚拟内存：建议打开虚拟内存，减少大块Block的申请导致的显存峰值过高，开启方式如下。值得注意的是，MS_ALLOC_CONF为kv格式，多个配置项需要一次性设置，以逗号分隔，避免多次设置被刷新。

    ```python
    export MS_ALLOC_CONF="enable_vmm:True"
    ```

- 手动开启GC：建议在每隔几百个step时，手动开启垃圾回收机制，它会检测不再使用的对象，并释放它们所占用的内存空间，如下所示。

    ```python
    import gc
    gc.collect()
    ```

### 显存优化配置建议

- 重计算：通过parallel_config中的recompute_config配置项使能，支持三种重计算的方式。一般并行场景为一维列表，vpp场景为二维列表，且只支持列表形式，列表中的元素代表vpp_rank/pp_rank下的ParallelTransformer实例需要重计算的层数。注：各类重计算加和的层数需要小于模型层数（select_comm_recompute_layers + full_recompute_layers <= num_layers, select_recompute_layers + full_recompute_layers <= num_layers）。
    - 完全重计算：full_recompute_list，完全重计算列表指定的ParallelTransformerLayer
    - 选择重计算：select_recompute_list，使用SA时会重计算attention部分，使用FA时，由于FA会自带重计算，此时会重计算MLP的激活部分，注意此时的激活函数需要写成nn.Cell的形式，否则不生效。
    - 通信重计算：select_comm_recompute_list，重计算SP并行的Allgather部分。

    ```yaml
    parallel_config:
      tensor_model_parallel_size: 2
      pipeline_model_parallel_size: 2
      num_layer_list: [2, 2]
      recompute_config:
        recompute: [1, 1]
        select_recompute: [1, 1]
        select_comm_recompute: [1, 1]
    ```

- 分布式优化器：在DP场景下，引入分布式优化器策略，该策略将重复的内存分配和计算任务分解，并借助高效通信机制进行信息交换，从而在不牺牲最终结果的前提下，显著降低内存占用与计算时间。

    ```yaml
    accelerate_config:
      parallelization:
        use_distributed_optimizer: True   # 是否启用分布式优化器
    ```

### 显存问题分析流程

如上述checklist排查和显存配置优化均完成后仍有显存不足的风险，则可根据下列流程分析具体细节。

1. 根据模型计算理论显存值，参考：[模型基础](https://3ms.huawei.com/km/blogs/details/15707187#preview_wps_15707187)
2. 打开流同步

    ```python
    ms.set_context(pynative_synchronize=True)
    ```

3. 打开显存抓取配置

    ```python
    export MS_ALLOC_CONF="memory_tracker:True"
    ```

4. 执行获取显存分析报告（二选一）
    - 缩小层数实际运行（推荐）
    - Dryrun模拟：export MS_SIMULATION_LEVEL=1（注：需要配套MindSpore 2.5版本）
5. 根据显存分析报告，重点分析未及时释放的显存，参考：[显存报告解析](https://wiki.huawei.com/domains/53421/wiki/72013/WIKI202412105373591?title=_37d45f57)

### 显存问题案例

[DiT自定义Cell显存分析](https://3ms.huawei.com/km/blogs/details/16639688)

## 参考资料

1. [模型FLOPs计算](https://jx.huawei.com/community/comgroup/postsDetails?postId=cc0e564acd794693bead67473145e94f)