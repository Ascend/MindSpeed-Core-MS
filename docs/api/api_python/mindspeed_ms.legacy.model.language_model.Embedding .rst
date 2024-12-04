mindspeed_ms.legacy.model.language_model.Embedding
==================================================

.. py:class:: mindspeed_ms.legacy.model.language_model.Embedding(hidden_size, vocab_size, max_sequence_length, embedding_dropout_prob, config, num_tokentypes=0, **kwargs)

    embedding层包含word embedding、position embedding和tokentypes embedding。

    参数:
        hidden_size (int): embedding层的隐藏状态大小.
        vocab_size (int): 词汇表大小.
        max_sequence_length (int): 序列的最大长度，用于position embedding。如果使用了position embedding，必须设置最大序列长度。
        embedding_dropout_prob (float): embedding层的dropout rate。
        config (TransformerConfig): Transformer模型的配置，详情请参考TransformerConfig类。
        num_tokentypes (int, optional): token-type embeddings的数量. 如果大于0，则使用tokentype嵌入。

    输入:
        - **input_ids** (Tensor) - int32类型的的输入索引, 形状为 :math:`(B, S)`。
        - **position_ids** (Tensor) - 用于position embedding的位置索引, 形状为 :math:`(B, S)`。
        - **tokentype_ids** (Tensor) - 用于区分不同类型标记（例如，在 BERT 中区分句子 A 和句子 B）的标记类型，形状为 :math:`(B, S)`。

    输出:
        - **embeddings** (Tensor)- embedding后的输出, 形状为 :math:`(B, S, H)`。

    异常:
        - **NotImplementedError** - 如果 `config.clone_scatter_output_in_embedding` 为 ``True``。
        - **RuntimeError** - 如果 `tokentype_ids` 不为 ``None`` 并且 `tokentype_embeddings` 为 ``None``.
            如果 `tokentype_ids` 为 ``None`` 并且 `tokentype_embeddings` 不为 ``None``.
