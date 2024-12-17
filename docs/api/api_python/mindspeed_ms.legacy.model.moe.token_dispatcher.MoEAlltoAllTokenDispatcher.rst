mindspeed_ms.legacy.model.moe.token_dispatcher.MoEAlltoAllTokenDispatcher
=========================================================================

.. py:class:: mindspeed_ms.legacy.model.moe.token_dispatcher.MoEAlltoAllTokenDispatcher(num_local_experts: int, local_expert_indices: List[int], config: TransformerConfig)

    在MoE架构中，MoEAlltoAllTokenDispatcher调度器负责将token令牌分配给各个专家进行处理，并将处理后的结果重新组合回原始的token顺序。

    参数:
        num_local_experts (int) - 表示当前 `rank` 有多少专家。
        local_expert_indices (List[int]) - 当前 `rank` 中专家的索引序号。
        config (TransformerConfig) - Transformer模型的配置，详情请参考TransformerConfig类。

    异常:
        - **ValueError** - 如果 `num_local_experts` 不大于 ``0`` 。
        - **ValueError** - 如果 `local_expert_indices` 的元素数量不等于 ``num_local_experts`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
