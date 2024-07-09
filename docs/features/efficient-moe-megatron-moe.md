# Efficient-MOE-Megatron-MOE
## 1. MoE 负载感知内存均衡算法

### 问题分析

MOE在预训练前期负载均衡aux_loss未起作用时，token在专家层的分配不均会导致全局内存负载不均衡问题，并引入大量碎片内存，导致显存波动巨大，并呈现逐步递增的趋势，大集群训练下更容易出现OOM。

### 优化方案

根据模型设定参数（DP/TP/SeqLength/NumExpert等），设定token分布不均的判定阈值，当超过该阈值后执行重计算，避免产生内存不均衡的激活值。

### 使用限制

1. 使用时**建议**开启`--moe-router-load-balancing-type aux_loss`，这样会使得训练过程中token分布快速趋向于平衡。
2. 开启全局重计算后该功能无效。

### 使用方法

- 必选参数：脚本中加入`--moe-adaptive-recompute-activation`即可。

- 可选参数：如果希望调节判定阈值，则修改`--moe-adaptive-recompute-activation-scale`即可，默认值为2.0，表示阈值为平均分配在每个专家上的序列的两倍长度。


## 2. MoE Group GEMM

### 问题分析

针对MoE单卡多专家计算，存在细碎的专家计算操作与通信，通过Group GEMM算子对多专家计算进行合并，提升MoE单卡多专家训练性能。

### 使用方法

设置`--moe-grouped-gemm`: 表示开启Group-GEMM计算。

## 效果说明

典型场景：

- EP变小导致单卡专家数量增大 & DeepSeek MoE专家数量较多等场景。
- DeepSeek MoE finegrained expert单个专家较小 & FFN规模不大 & TP变大导致单卡切分的计算变小。

1. 随着FFN规模提升，计算不再细碎，单专家计算效率提升，Group GEMM收益变小。

表1：grok模型FFN大小和性能加速对比
|ffn_hidden_size| 32768 | 16384| 8192| 4096|
|--|--|--|--|--|
|baseline|2280|1780|1537|1446|
|GeMM|2416|1719|1448|1331|
|性能提升|-5.30%|3.53%|6.12%|8.60%|


2. TP越大，EP越小，收益更大。
   
表2：Mixtral8*7B模型配置不同性能收益
|配置| tp4 ep2 16expert | tp4 ep2 8expert | tp2 ep4 16expert| tp2 ep4 8expert|
|--|--|--|--|--|
|baseline|27969|20127|11976|13981|
|GeMM|19415|17361|11049|14290|
|性能提升|44.06%|17.93%|8.39%|-2.19%|
