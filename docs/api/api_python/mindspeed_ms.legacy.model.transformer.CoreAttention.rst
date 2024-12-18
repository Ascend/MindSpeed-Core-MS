mindspeed_ms.legacy.model.transformer.CoreAttention
===================================================

.. py:class:: mindspeed_ms.legacy.model.transformer.CoreAttention(layer_number, config, attn_mask_type=AttnMaskType.padding)

    核心注意力机制，用于计算查询、键和值层之间的注意力权重和上下文表示。

    参数：
        - **layer_number** (int) - 该transformer层在整个transformer块中的索引。
        - **config** (dict) - Transformer模型的配置，详情请参考TransformerConfig类。
        - **attn_mask_type** (int，可选) - Attention mask type，支持[AttnMaskType::padding = 1, AttnMaskType::causal = 2]。默认为 ``1`` 。

    输入：
        - **query_layer** (Tensor) - 查询层。形状为 :math:`(S, B, N, D)` 。
        - **key_layer** (Tensor) - 键层。形状为 :math:`(S, B, N, D)` 。
        - **value_layer** (Tensor) - 值层。形状为 :math:`(S, B, N, D)` 。
        - **attention_mask** (Tensor) - 注意力掩码。形状为 :math:`(B, S_q, S_k)` 。

    输出：
        - **context_layer** (Tensor) - 形状为 :math:`(S, B, H)` 。

    异常：
        - **NotImplementedError** - 如果 `config` 中的 `masked_softmax_fusion` 为 ``True`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
