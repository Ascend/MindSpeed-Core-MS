# Alibi 位置编码

## 问题分析

当大模型在训练和预测时的输入长度不一致时，模型的泛化能力会下降。若外推能力不佳，大模型在处理长文本或多轮对话时的效果就会受到限制。正弦位置编码的外推能力比较弱，RoPE（Rotary Position Embedding）的外推能力有一定提高但仍然有限。

## 解决方案

支持Alibi位置编码，提高模型外推能力。

### 解决思路:

Alibi算法给attention score添加了一个预设的线性偏置矩阵（如下图所示），使模型能够理解输入之间的相对位置关系。由于位置信息直接作用于attention score上，位置差异性被突出，使模型具有较强的外推能力。
<p align="center"> <img src="../../sources/images/alibi.png" height="180px" width="400px"></p>

[原文链接](https://arxiv.org/pdf/2108.12409)


## 使用方法

设置`--position-embedding-type alibi`即可调用该算法。

## 使用效果

模型外推能力提高。
