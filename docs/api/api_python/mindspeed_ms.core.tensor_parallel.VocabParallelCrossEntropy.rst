mindspeed_ms.core.tensor_parallel.VocabParallelCrossEntropy
===========================================================

.. py:class:: mindspeed_ms.core.tensor_parallel.VocabParallelCrossEntropy(*args, **kwargs)

    交叉熵损失函数的并行接口。

    参数：
        - **args** (tuple) - 位置参数。
        - **kwargs** (dict) - 其他输入。

    输入：
        - **vocab_parallel_logits** (Tensor) - 主干网络的输出。形状为 :math:`(N, C)` 的张量。数据类型必须为float16或float32。
        - **target** (Tensor) - 样本的真值。形状为 :math:`(N,)` 。
        - **label_smoothing** (float，可选) - 平滑因子，必须在范围 ``[0.0, 1.0)`` 内。默认值： ``0.0`` 。

    输出：
        - **loss** (Tensor) - 对应的交叉熵损失。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
