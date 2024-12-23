mindspeed_ms.legacy.model.moe.experts.SequentialMLP
===================================================

.. py:class:: mindspeed_ms.legacy.model.moe.experts.SequentialMLP(num_local_experts: int, config: TransformerConfig, submodules=None)

    定义SequentialMLP模块。

    参数：
        - **num_local_experts** (int) - 局部专家的数量。
        - **config** (TransformerConfig) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **submodules** (MLPSubmodules) - 线性全连接层的类型。保留参数，目前没有使用。

    输入：
        - **permuted_local_hidden_states** (Tensor) - 局部专家的隐藏层排列输入。
        - **token_per_expert** (Tensor) - 每个专家的token数量。

    输出：
        两个张量的元组。

        - **output_local** (Tensor) - 局部专家的输出。
        - **output_bias_local** (Tensor) - 局部专家输出偏置。目前没有使用。默认返回 ``None`` 。

    异常：
        - **NotImplementedError** - 如果 `submodules` 不为 ``None`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。