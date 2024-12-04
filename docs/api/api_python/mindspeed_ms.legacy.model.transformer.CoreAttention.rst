mindspeed_ms.legacy.model.transformer.CoreAttention
===================================================

.. py:class:: mindspeed_ms.legacy.model.transformer.CoreAttention(layer_number, config, attn_mask_type=AttnMaskType.padding)

    计算注意力得分。基于query、key、and、value layers和attention mask，计算context layer。

    参数:
        - **layer_number** (int): 该transformer层在整个transformer块中的索引。
        - **config** (dict): Transformer模型的配置，详情请参考TransformerConfig类。
        - **attn_mask_type** (int): Attention mask type，支持[AttnMaskType::padding = 1, AttnMaskType::causal = 2]。默认为 ``1``。

    输入:
        - **query_layer** (Tensor) - 形状为 :math:`(S, B, N, D)`。
        - **key_layer** (Tensor) - 形状为 :math:`(S, B, N, D)`。
        - **value_layer** (Tensor) - 形状为 :math:`(S, B, N, D)`。
        - **attention_mask** (Tensor) - 形状为 :math:`(B, S_q, S_k)`。

    输出:
        - **context_layer** (Tensor) - 形状为 :math:`(S, B, H)`。

    异常:
        - **NotImplementedError** - 如果 `config` 中的 `masked_softmax_fusion` 为 ``True``。
