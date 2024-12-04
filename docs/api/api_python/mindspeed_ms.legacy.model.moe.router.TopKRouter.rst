mindspeed_ms.legacy.model.moe.router.TopKRouter
===============================================

.. py:class:: mindspeed_ms.legacy.model.moe.router.TopKRouter(config: TransformerConfig)

    TopK router。

    参数:
        config (TransformerConfig): Transformer模型的配置，详情请参考TransformerConfig类。

    输入:
        - **input** (Tensor) - 输入张量。

    输出:
        2个张量组成的Tuple

        - **scores** (Tensor) - 负载均衡后的概率张量。
        - **indices** (Tensor) - 经过 top-k 选择后的索引张量。

    异常:
        - **NotImplementedError** - 如果 `self.routing_type` 为 ``sinkhorn``。
        - **ValueError** - 如果 `self.routing_type` 不为 ``None``。
