mindspeed_ms.training.loss_func.LossWithMask
============================================

.. py:class:: mindspeed_ms.training.loss_func.LossWithMask(loss_func, *args, **kwargs)

    使用掩码和均值计算损失。

    参数：
        - **loss_func** (Function) - 损失函数。
        - **args** (Tuple) - 位置参数。
        - **kwargs** (dict) - 其他输入。

    输入：
        - **logits** (Tensor) - 主干网络的输出。形状为 :math:`(N, C)` 的张量。数据类型必须为float16或float32。
        - **label** (Tensor) -  样本的真值。形状为 :math:`(N, )` 或与 `logits` 相同维度的张量。
        - **input_mask** (Tensor) - `input_mask`表示是否有填充的输入，对于填充的输入，它不会被计算到损失中。形状为 :math:`(N, )` 的张量。

    返回：
        对应的交叉熵。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
