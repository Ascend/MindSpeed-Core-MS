mindspeed_ms.legacy.model.ParallelTransformerLayer
==================================================

.. py:class:: mindspeed_ms.legacy.model.ParallelTransformerLayer(config, layer_number, layer_type=LayerType.encoder, self_attn_mask_type=AttnMaskType.padding, drop_path_rate=0.0)

    单独的一层transformer。它结合了归一化、注意力、交叉注意力和MLP来处理输入隐藏状态。

    参数：
        - **config** (dict) - 一个配置字典，提供了transformer层的各种参数配置。
        - **layer_number** (int) - 该transformer层在整个transformer块中的索引。
        - **layer_type** (int，可选) - layer类型。支持1为encoder，2为decoder，3为retro_encoder，4为retro_decoder，5为retro_decoder_with_retriever, 默认值： ``LayerType.encoder`` 。
        - **self_attn_mask_type** (int，可选) - 注意力mask类型。支持1为padding，2为causal，默认值： ``AttnMaskType.padding`` 。
        - **drop_path_rate** (float，可选) - drop_path rate。当前不支持该参数大于0，默认值： ``0.0`` 。

    输入：
        - **hidden_states** (Tensor) - 隐藏层状态张量，形状为 :math:`(B, S, H)` 。
        - **attention_mask** (Tensor) - attention掩码矩阵。
        - **encoder_output** (Tensor，可选) - 用于交叉注意力的编码器输出张量，当前不支持该参数。默认值： ``None`` 。
        - **enc_dec_attn_mask** (Tensor，可选) - 编码器-解码器注意力mask张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_input** (Tensor，可选) - 检索输入张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_output** (Tensor，可选) - 检索输出张量，当前不支持该参数。默认值： ``None`` 。
        - **retriever_attn_mask** (Tensor，可选) - 检索注意力mask张量，当前不支持该参数。默认值： ``None`` 。
        - **inference_params** (Tensor，可选) - 推理参数的张量，当前不支持该参数。默认值： ``None`` 。
        - **rotary_pos_emb** (Tensor，可选) - 旋转位置嵌入张量。默认值： ``None`` 。

    输出：
        - **output** (Tensor) - 输出张量形状为 :math:`(B, S, H)` 。

    异常：
        - **NotImplementedError** - 如果 `config` 中的 `bias_dropout_fusion` 是 ``True`` 。
        - **NotImplementedError** - 如果 `drop_path_rate` 大于 0。
        - **NotImplementedError** - 如果 `config` 中的 `retro_add_retriever` 是 ``True`` 。
        - **NotImplementedError** - 如果 `encoder_output` 、 `enc_dec_attn_mask` 、 `retriever_input` 、 `retriever_output` 、 `retriever_attn_mask` 或 `inference_params` 不是 ``None`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好通信环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
