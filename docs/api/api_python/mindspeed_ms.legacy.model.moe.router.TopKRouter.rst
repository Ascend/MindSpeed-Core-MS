mindspeed_ms.legacy.model.moe.router.TopKRouter
===============================================

.. py:class:: mindspeed_ms.legacy.model.moe.router.TopKRouter(config: TransformerConfig)

    Top-K路由，负责根据输入数据计算分数，并选择Top-K个专家来处理输入数据。

    参数：
        - **config** (TransformerConfig) - Transformer模型的配置，详情请参考TransformerConfig类。

    输入：
        - **input** (Tensor) - 输入张量。

    输出：
        两个张量组成的元组。

        - **scores** (Tensor) - 负载均衡后的概率张量。
        - **indices** (Tensor) - 经过 top-k 选择后的索引张量。

    异常：
        - **NotImplementedError** - 如果 `moe_config.moe_router_load_balancing_type` 为 ``sinkhorn`` 。
        - **ValueError** - 如果 `moe_config.moe_router_load_balancing_type` 不是 `sinkhorn` ，不是 `aux_loss` ，也不是 `none` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
