mindspeed_ms.legacy.model.ParallelTransformer
=============================================

.. py:class:: mindspeed_ms.legacy.model.ParallelTransformer(config, model_type, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding, post_norm=True, pre_process=False, post_process=False, drop_path_rate=0.0)

    Transformer模块。它由多个单独的transformer层组成，可以处理各种配置和处理步骤。

    参数：
        - **config** (dict) - 一个配置字典，提供了并行transformer层的各种参数配置。
        - **model_type** (int) - model类型。支持1为encoder_or_decoder，2为encoder_and_decoder，3为retro_encoder，4为retro_decoder。
        - **layer_type** (int，可选) - layer类型。支持1为encoder，2为decoder，3为retro_encoder，4为retro_decoder，5为retro_decoder_with_retriever, 默认值： ``LayerType.encoder`` 。
        - **self_attn_mask_type** (int，可选) - 注意力mask类型。支持1为padding，2为causal，默认值： ``AttnMaskType.padding`` 。
        - **post_norm** (bool，可选) - 是否在转换器块的末尾插入归一化层。默认值： ``True`` 。
        - **pre_process** (bool，可选) - 使用流水线并行时，表明它是否是第一阶段。默认值： ``False`` 。
        - **post_process** (bool，可选) - 使用流水线并行时，表明它是否是最后一个阶段。默认值： ``False`` 。
        - **drop_path_rate** (float，可选) - 丢弃率。当前不支持该参数大于0，默认值： ``0.0`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)` 。
        - **attention_mask** (Tensor) - 注意力掩码张量。
        - **encoder_output** (Tensor，可选) - 用于交叉注意力的编码器输出张量，当前不支持该参数。默认值： ``None`` 。
        - **enc_dec_attn_mask** (Tensor，可选) - 编码器-解码器注意力mask张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_input** (Tensor，可选) - 检索输入张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_output** (Tensor，可选) - 检索输出张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_attn_mask** (Tensor，可选) - 检索注意力mask张量，当前不支持该参数。默认值： ``None`` 。
        - **inference_params** (Tensor，可选) - 推理参数的张量，当前不支持该参数。默认值： ``None`` 。
        - **rotary_pos_emb** (Tensor，可选) - 旋转位置嵌入张量。默认值： ``None`` 。

    输出：
        - **hidden_states** (Tensor) - 输出张量形状为 :math:`(B, S, H)` 。

    异常：
        - **NotImplementedError** - 如果 `drop_path_rate` 大于 ``0`` 。
        - **NotImplementedError** - 如果 `config` 中的 `distribute_saved_activations` 是 ``True`` 并且 `config` 中的 `sequence_parallel` 是 ``False`` 。
        - **NotImplementedError** - 如果 `config` 中的 `transformer_impl` 是 ``transformer_engine`` 。
        - **NotImplementedError** - 如果 `config` 中的 `fp8` 不是 ``None`` 。
        - **NotImplementedError** - 如果 `config` 中的 `retro_add_retriever` 是 ``True`` 。
        - **NotImplementedError** - 如果 `model_type` 是 ``3`` 或 ``4`` 。
        - **NotImplementedError** - 如果 `encoder_output` 、`enc_dec_attn_mask` 、 `retriever_input` 、 `retriever_output` 、 `retriever_attn_mask` 或 `inference_params` 不是 ``None`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
