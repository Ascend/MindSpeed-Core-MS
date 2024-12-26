mindspeed_ms.legacy.model.ParallelAttention
===========================================

.. py:class:: mindspeed_ms.legacy.model.ParallelAttention(config, layer_number, attention_type=AttnType.self_attn, attn_mask_type=AttnMaskType.padding)

    该类表示并行注意力机制。它可以处理不同的注意力类型，并且可以使用各种参数进行配置。

    参数：
        - **config** (dict) - Transformer模型配置。详情请参考TransformerConfig类。
        - **layer_number** (int) - 该transformer层在整个transformer块中的索引。
        - **attention_type** (int，可选) - 注意力类型。支持1为self_attn，2为cross_attn，默认值： ``AttnType.self_attn`` 。
        - **attn_mask_type** (int，可选) - 注意力掩码类型。支持1为padding，2为causal，默认值： ``AttnMaskType.padding`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)` 。
        - **attention_mask** (Tensor) - attention掩码矩阵，形状为 :math:`(B, N, S_q, S_k)` 。
        - **encoder_output** (Tensor，可选) - 用于交叉注意力的编码器输出张量。默认值： ``None`` 。
        - **inference_params** (Tensor，可选) - 推理参数的张量，当前不支持该参数。默认值： ``None`` 。
        - **rotary_pos_emb** (Tensor，可选) - 旋转位置嵌入张量。默认值： ``None`` 。

    输出：
        - **output** (Tensor) - 输出张量形状为 :math:`(B, S, H)` 。
        - **bias** (Tensor) - 可训练的偏置参数。

    异常：
        - **NotImplementedError** - 如果使用了flash attention，但是 `attention_type` 是 `AttnType.self_attn` 。
        - **ValueError** - 如果 `group_query_attention` 是 `True` 但是 `num_query_groups` 不能被 `tp_group_size` 整除。
        - **ValueError** - 如果 `attention_type` 既不是 ``AttnType::self_attn`` 也不是 ``AttnType::cross_attn`` 。
        - **NotImplementedError** - 如果 `attention_type` 是 2 并且 `config` 中的 `group_query_attention` 是 ``True`` 。
        - **ValueError** - 如果 `config` 中的 `hidden_size` 不等于 `config` 中的 `kv_hidden_size` 并且 `attention_type` 是 2。
        - **NotImplementedError** - 如果 `get_context_parallel_world_size() > 1` 并且 `args.context_parallel_algo` 为 `ulysses_cp_algo` 并且没有使用flash attention。
        - **NotImplementedError** - 如果 `inference_params` 不是 ``None`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
