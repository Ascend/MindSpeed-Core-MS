mindspeed_ms.legacy.model.module.Module
=======================================

.. py:class:: mindspeed_ms.legacy.model.module.Module(config=None, share_embeddings_and_output_weights=True, **kwargs)

    具有流水线支持的特定扩展单元。

    参数：
        - **config** (dict，可选) - 模型的配置。如果它不是 ``None`` ， `self.pre_process` 和 `self.post_process` 将根据流水线阶段进行设置。默认值： ``None`` 。
        - **share_embeddings_and_output_weights** (bool，可选) - 决定是否共享嵌入层和输出层权重。如果不是 ``True`` ，将无法调用 `shared_embedding_or_output_weight()` 和 `initialize_word_embeddings()` 。默认值： ``True`` 。
        - **kwargs** (dict) - 额外的关键字配置参数。

    异常：
        - **RuntimeError** - 在流水线的一个阶段中设置了多个权重的 `'share'` 属性。
        - **RuntimeError** - 模型中有一个权重具有 `'share'` 属性，但参数共享要求在第一阶段和最后阶段分别有两个权重具有 `'share'` 属性。
        - **RuntimeError** - 在调用 `shared_embedding_or_output_weight()` 时 `share_embeddings_and_output_weights` 不为 ``True`` 。
        - **RuntimeError** - 在调用 `initialize_word_embeddings()` 时 `share_embeddings_and_output_weights` 不为 ``True`` 。
        - **ValueError** - 如果是最后一阶段（后处理），但是权重之和不为 ``0`` 。
