mindspeed_ms.legacy.model.ParallelMLP
=====================================

.. py:class:: mindspeed_ms.legacy.model.ParallelMLP(config, is_expert=False)

    并行前馈模块实现。

    参数：
        - **config** (TransformerConfig) - transformer模型的config。
        - **is_expert** (bool，可选) - 指定这个block是否是专家. 默认值: ``False`` 。

    输入：
        - **hidden_states** (Tensor) - 一个形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量。

    输出：
        - **output** (Tensor) - 一个形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
