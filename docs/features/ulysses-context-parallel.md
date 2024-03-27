# Ulysses长序列并行

## 问题分析

从生成性AI到科研模型，长序列训练正在变得非常重要。 在生成性AI领域，会话式AI、长文档摘要和视频生成等任务都需要在空间和时间层面对长上下文进行推理。 同样，章节和书籍级别的摘要（数万甚至数十万字）在会话式AI和摘要任务中也非常重要。现有的数据、张量和流水线等并行方法无法解决序列维度的扩展问题。

## 解决方案

支持ulysses长序列并行方案，以此解决序列维度扩展问题。

### 解决思路:

DeepSpeed-Ulysses将各个样本在序列维度上分割给参与的GPU。然后，在attention计算之前，它对已分割的查询(Q)、键(K)和值(V)执行all-to-all通信操作，以使每个GPU接收完整的序列，但仅用于注意力头的非重叠子集。这使得参与的GPU可以并行计算不同的注意力头。最后，DeepSpeed-Ulysses还使用另一个all-to-all来在注意力头上收集结果，同时重新在序列维度上进行分区。

## 使用场景

num_head既要能被tp_size整除，也要能被cp_size整除。

## 使用方法

设置`--context-parallel-size`，默认为1，根据用户需求配置。
设置`--context-parallel-algo`，可选项为`ulysses_cp_algo`或者`megatron_cp_algo`，默认`ulysses_cp_algo`。

## 使用效果

可支持长序列并行。

