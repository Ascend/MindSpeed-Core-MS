mindspeed_ms.legacy.model.TransformerLanguageModel
==================================================

.. py:class:: mindspeed_ms.legacy.model.TransformerLanguageModel(config, encoder_attn_mask_type, num_tokentypes=0, add_encoder=True, add_decoder=False, decoder_attn_mask_type=AttnMaskType.causal, add_pooler=False, pre_process=True, post_process=True, visual_encoder=None, **kwargs)

    Transformer语言模型。

    参数：
        - **config** (TransformerConfig) - Transformer模型配置，包括初始化函数和并行参数配置等。
        - **encoder_attn_mask_type** (int) - 编码器注意力掩码类型。
        - **num_tokentypes** (int，可选) - 如果大于0，则使用tokentype嵌入。默认值：``0`` 。
        - **add_encoder** (bool，可选) - 如果为 ``True`` ，使用编码器。默认值：``True`` 。
        - **add_decoder** (bool，可选) - 如果为 ``True`` ，使用解码器。默认值：``False`` 。
        - **decoder_attn_mask_type** (int，可选) - 解码器注意力掩码类型。默认值：``AttnMaskType.causal`` 。
        - **add_pooler** (bool，可选) - 如果为 ``True`` ，使用池化层。默认值：``False`` 。
        - **pre_process** (bool，可选) - 使用流水线并行时，标记它是否为第一阶段。默认值：``True`` 。
        - **post_process** (bool，可选) - 使用流水线并行时，标记它是否为最后的阶段。默认值：``True`` 。
        - **visual_encoder** (nn.Cell，可选) - 视觉编码器。默认值：``None`` 。
        - **kwargs** (dict) - 其他输入。

    输入：
        - **enc_input_ids** (Tensor) - 编码器输入索引。形状为 :math:`(B, S)` 。
        - **enc_position_ids** (Tensor) - 编码器位置偏移量。形状为 :math:`(B, S)` 。
        - **enc_attn_mask** (Tensor) - 编码器注意力掩码。形状为 :math:`(B, S)` 。
        - **dec_input_ids** (Tensor，可选) - 解码器输入索引。形状为 :math:`(B, S)` 。默认值： ``None`` 。
        - **dec_position_ids** (Tensor，可选) - 解码器输入位置索引。形状为 :math:`(B, S)` 。默认值： ``None`` 。
        - **dec_attn_mask** (Tensor，可选) - 解码器注意力掩码。形状为 :math:`(B, S)` 。默认值： ``None`` 。
        - **retriever_input_ids** (Tensor，可选) - 检索器输入标记索引。默认值： ``None`` 。
        - **retriever_position_ids** (Tensor，可选) - 检索器输入位置索引。默认值： ``None`` 。
        - **retriever_attn_mask** (Tensor，可选) - 检索器注意力掩码，用于控制在检索器中计算注意力时的注意范围。默认值： ``None`` 。
        - **enc_dec_attn_mask** (Tensor，可选) - 编码器-解码器注意力掩码，用于在编码器和解码器之间计算注意力时使用。默认值： ``None`` 。
        - **tokentype_ids** (Tensor，可选) - 给模型输入的标记类型索引列表。形状为 :math:`(B, S)` 。默认值： ``None`` 。
        - **inference_params** (InferenceParams，可选) - 推理参数，用于在推理过程中指定特定设置，如最大生成长度、最大批处理大小等。默认值： ``None`` 。
        - **pooling_sequence_index** (int，可选) - 池化序列索引。默认值： ``0`` 。
        - **enc_hidden_states** (Tensor，可选) - 编码器隐藏层。默认值： ``None`` 。
        - **output_enc_hidden** (bool，可选) - 是否输出编码器隐藏层。默认值： ``False`` 。
        - **input_image** (Tensor，可选) - 输入图像的张量。形状为 :math:`(N, C_{in}, H_{in}, W_{in})` 或 :math:`(N, H_{in}, W_{in}, C_{in}, )` 。默认值： ``None`` 。
        - **delimiter_position** (Tensor，可选) - 分隔符位置张量。形状为 :math:`(B, N)` ，其中 :math:`N` 表示分隔符数量。默认值： ``None`` 。
        - **image_embedding** (Tensor，可选) - 图像嵌入张量，维度依赖于图像嵌入的维数。默认值： ``None`` 。

    输出：
        - **encoder_output** (Tensor) - 形状为 :math:`(B, S, H)` 或 :math:`(S, B, H)` 的张量。

    异常：
        - **ValueError** - 如果 `config.untie_embeddings_and_output_weights` 且 `add_decoder` 为 ``True`` 。
        - **RuntimeError** - 如果 `input_tensor` 长度为 `1` 。
        - **NotImplementedError** - 如果 `config.retro_add_retriever` 为 ``True`` 。
        - **NotImplementedError** - 如果 `visual_encoder` 或者 `add_decoder` 为 ``True`` 。
        - **NotImplementedError** - 如果 `dec_input_ids` 、 `dec_position_ids` 、 `dec_attn_mask` 、 `retriever_input_ids` 、 `retriever_position_ids` 、 `retriever_attn_mask` 、 `enc_dec_attn_mask` 、 `input_image` 、 `delimiter_position` 或者 `image_embedding` 不为 ``None`` 。
        - **NotImplementedError** - 如果 `output_enc_hidden` 为 ``True`` 。

    样例：

    .. note::
        - 运行样例之前，需要配置好环境变量。
        - 针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/msrun_launcher.html>`_ 。
