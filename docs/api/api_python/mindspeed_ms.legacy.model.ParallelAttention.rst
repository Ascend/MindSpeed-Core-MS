mindspeed_ms.legacy.model.ParallelAttention
===========================================

.. py:class:: mindspeed_ms.legacy.model.ParallelAttention(config, layer_number, attention_type=1, attn_mask_type=1)

    并行自注意力机制类。输入 :math:`(B, S, H)` 的张量，返回相同形状的输出。

    参数：
        - **config** (dict) - 一个配置字典，提供了并行注意力机制的各种设置。
        - **layer_number** (int) - 该transformer层在整个transformer块中的索引。
        - **attention_type** (int) - 注意力类型。支持1为self_attn，2为cross_attn，默认值： ``1`` 。
        - **attn_mask_type** (int) - 注意力mask类型。支持1为padding，2为causal，默认值： ``1`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)`.
        - **attention_mask** (Tensor) - attention掩码矩阵，形状为 :math:`(B, N, S_q, S_k)`。
        - **encoder_output** (Tensor) - 用于交叉注意力的编码器输出张量。默认值： ``None``。
        - **inference_params** (Tensor) - 推理参数的张量，当前不支持该参数。默认值： ``None``。
        - **rotary_pos_emb** (Tensor) - 旋转位置嵌入张量。默认值： ``None``。

    输出：
        - **output** (Tensor) - 输出张量形状为 :math:`(B, S, H)`。
        - **bias** (Tensor) - 可训练的偏置参数。

    异常：
        - **NotImplementedError** - 如果使用了flash attention，但是 `attention_type` 是 `AttnType.self_attn` 。
        - **ValueError** - 如果 `group_query_attention` 是 `True` 但是 `num_query_groups` 不能被 `tp_group_size` 整除。
        - **ValueError** - 如果 `attention_type` 既不是 1 也不是 2。
        - **NotImplementedError** - 如果 `attention_type` 是 2 并且 `config` 中的 `group_query_attention` 是 true。
        - **ValueError** - 如果 `config` 中的 `hidden_size` 不等于 `config` 中的 `kv_hidden_size` 并且 `attention_type` 是 2。
        - **NotImplementedError** - 如果 `get_context_parallel_world_size() > 1` 并且 `args.context_parallel_algo` 为 `ulysses_cp_algo` 并且没有使用flash attention。
        - **NotImplementedError** - 如果 `inference_params` 不是 None。
