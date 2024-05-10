# 激活函数重计算

## 问题分析

在大模型训练过程中，如gelu激活函数会产生大量的数据，但本身计算量很小。激活函数产生的数据会占据大量内存。

## 解决方案

此时进行激活函数的重计算可以在性能劣化极少的代价下，减少内存占用。
尤其实在诸如反向传播与梯度计算等操作时，通过仅存储必要的中间结果来节省资源。

## 解决思路

设计一种传入激活函数进行重计算的机制，在合适的时机，丢弃重计算模块输出的物理存储，保留逻辑视图。在反向时，利用传入的激活函数重新进行计算，得到结果。

## 使用场景

主要用于训练场景，用户内存不足或要节省内存时

## 使用方法

脚本中添加：--recompute-activation-function 可开启激活函数重计算

添加：--recompute-activation-function-num-layers ${num} 可指定激活函数重计算的层数

激活函数重计算可以与全重计算同时开启：

1.同时开启时，仅支持 --recompute-method 为 block

2.同时开启时，会按照指定的全重计算和激活函数重计算的层数做各自类型的重计算，即不会有一层既做全重计算又做激活函数重计算。

## 扩展使用

在设计传入激活函数进行全重计算时，引入的 CheckpointWithoutOutput 类具有通用进行函数重计算的机制。
不仅可以对激活函数进行重计算，也可以对任何自定义的函数进行重计算。

此处提供一个示例，可以灵活使用 CheckpointWithoutOutput 来对自定义的函数进行重计算：

```python
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput


class Custom_module(torch.nn.Module):
    def __init__(self):
        ......

    def forward(self, input):
        self.activation_checkpoint_manager = CheckpointWithoutOutput()
        function_output = self.activation_checkpoint_manager.checkpoint(self.custom_function, False, function_input1, function_input2, ...)
        ...(after used output)
        self.activation_checkpoint_manager.discard_output()
        module_output.register_hook(self.activation_checkpoint_manager.recompute)

        return module_output
```