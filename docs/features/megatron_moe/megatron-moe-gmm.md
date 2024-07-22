# Megatron MoE Grouped GeMM

## 问题分析

针对MoE单卡多专家计算，存在细碎的专家计算操作与通信，通过Grouped GeMM算子对多专家计算进行合并，提升MoE单卡多专家训练性能。

## 解决方案

通过调用 gmm 融合算子，对多个专家计算进行融合，达到加速效果。

## 使用方法

设置`--moe-grouped-gemm`: 表示开启Grouped GeMM计算。

## 效果说明

典型场景：

- EP变小导致单卡专家数量增大 & DeepSeek MoE专家数量较多等场景。
- DeepSeek MoE finegrained expert单个专家较小 & FFN规模不大 & TP变大导致单卡切分的计算变小。

1. 随着FFN规模提升，计算不再细碎，单专家计算效率提升，Grouped GeMM 收益变小。

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
