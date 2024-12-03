mindspeed_ms.legacy.model.ParallelMLP
=====================================

.. py:class:: mindspeed_ms.legacy.model.ParallelMLP(config, is_expert=False)

    并行前馈模块实现。

    参数:
        - **config** (TransformerConfig) - transformer模型的config。
        - **is_expert** (bool) - 指定这个block是否是专家. 默认: ``False``。

    输入:
        - **hidden_states** (Tensor) - 一个形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量。

    输出:
        - **output** (Tensor) - 一个形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量。
