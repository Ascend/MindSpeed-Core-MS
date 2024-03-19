# 计算通信并行 CoC (Communication Over Computation)

## 问题分析

大模型训练过程中，其ColumnParallelLinear和RowParallelLinear部分的前方向均存在相互毗邻、顺序依赖的计算通信组合，计算为Matmul，而通信则为AllReduce（不开启序列并行）或AllGather和ReduceScatter（开启序列并行）。这些计算通信的组合因为存在顺序依赖（即后一个的输入是前一个输出），常常被串行执行，但这时候计算和通信流都存在一定的空闲等待时间，该过程的执行效率没有被最大化。

## 解决方案

通过将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖。

### 解决思路

#### 脚本测实现
将张量进行进一步切分（2/4/8份），通过Python脚本的方式实现每个子tensor之间计算和通信的并行，从而增大计算和通信流的利用率；


#### 融合算子实现
基于MTE远端内存访问能力，以融合大Kernel方式在算子实现的内部将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖；

## 使用场景
该特性目前主要用于训练场景，当Attention模块和MLP模块串行执行且计算通信存在顺序依赖与位置毗邻关系时适用。

使用脚本测实现时，对Matmul左矩阵的m轴有一定要求，必须是切分数（2/4/8）的倍数，且不适用于计算与通信片段耗时相差较大的情况。需要注意的是，脚本测实现在切分矩阵、切分数量较大时，容易出现host bound问题，从而不能得到预期的收益。支持ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER三个通信场景，支持灵活设置先通信或先计算。

对于计算通信融合算子，目前已支持：
1. MATMUL_ALL_REDUCE融合算子（先计算后通信）及其确定性计算；
2. MATMUL_REDUCE_SCATTER融合算子（先计算后通信）及其确定性计算；
3. ALL_GATHER_MATMUL, ALL_GATHER_MATMUL_V2融合算子（先通信后计算）（V2版本接口支持ALL_GATHER中间结果获取）；
4. 量化场景：MATMUL_ALL_REDUCE融合算子支持fp16格式的w8A16伪量化，粒度包含per tensor / per channel / per group；

## 使用方法 —— 在ModelLink中进行整网训练

计算通信并行优化算法通过在ModelLink的训练脚本中配置环境变量来进行使能，需要安装ascendspeed。

- 如果不想使用计算通信并行，进行如下设置：
```shell
export CC_PARALLEL_NUM=1
export USE_CC_FUSED_KERNEL=0
```

- 如果想使用脚本测计算通信并行，进行如下设置：
```shell
export CC_PARALLEL_NUM=2 # 或者4，或者8
export USE_CC_FUSED_KERNEL=0
```

- 如果想使用计算通信并行融合算子，则需要进行如下设置：
```shell
export USE_CC_FUSED_KERNE=1
```

融合算子的环境变量拥有更高优先级，即当 CC_PARALLEL_NUM > 1 且 USE_CC_FUSED_KERNEL = 1 时，前者不会生效。

注意：**计算通信并行融合算子需要安装ATB后才能使用**！如果出现报错信息“找不到libatb.so”，则需要运行如下命令（将YOUR_ATB_PATH换成您的ATB目录）：
```shell
export LD_LIBRARY_PATH+=":YOUR_ATB_PATH/ascend-transformer-boost/output/atb/lib"
```

## 使用效果

在BLOOM 7B模型中获得端到端性能收益约3.20%，在BLOOM 176B模型中获得端到端性能收益约5.47%，在LLAMA2 70B模型中获得端到端性能收益约7.85%。精度相对误差控制在2%的范围内。

## 计算通信并行接口设置

在安装ascendspeed后，可以用过如下方式获取计算通信并行融合算子的调用接口。

```python
from ascendspeed.ops.lcal_functional import coc_ops
```

然后通过如下方式使用各个计算通信并行融合算子。

1. MATMUL_ALL_REDUCE

```python
coc_ops.matmul_all_reduce(input1, input2, output, bias)
```

其中input1, input2分别为左右矩阵，output为输出矩阵（需要提前申请内存空间并作为入参传入接口），bias为偏置(shape为[n]或者[1,n])，如果不需要则为None。当前只支持input2的转置，即input1的shape必须为[m,k]，而input2的shape可以为[k,n]，也可以为[n,k]。

2. ALL_GATHER_MATMUL

```python
coc_ops.all_gather_matmul(input1, input2, output, bias)
```

接口含义如上。

3. ALL_GATHER_MATMUL_V2

```python
coc_ops.all_gather_matmul_v2(input1, input2, output, comm_output, bias)
```

接口含义如上。comm_output为all_gather的结果，即通信结果，需要提前申请内存空间并作为入参传入接口。

4. MATMUL_REDUCE_SCATTER

```python
coc_ops.matmul_reduce_scatter(input1, input2, output, bias)
```

接口含义如上。

## 注意事项：

1. 计算通信并行优化特性不能与TP重计算通信优化特性同时使用，因为存在机制上的冲突；
