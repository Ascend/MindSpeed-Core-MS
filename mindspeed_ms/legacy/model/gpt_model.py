
"""GPT model."""

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.legacy.model.enums import AttnMaskType
from mindspeed_ms.legacy.model.language_model import get_language_model
from mindspeed_ms.legacy.model.transformer import ParallelLMLogits
from mindspeed_ms.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from mindspeed_ms.training.loss_func import LossWithMask
from mindspeed_ms.legacy.model.eos_mask import EosMask
from mindspeed_ms.core.parallel_state import get_context_parallel_rank, get_context_parallel_world_size

def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy, loss_mask):
    """ gpt model post process forward """
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        # [s b h] -> [b s h]
        return output.swapaxes(0, 1).contiguous()

    # [s b] -> [b s]
    output = output.swapaxes(0, 1).contiguous()

    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels, loss_mask)
    else:
        loss = loss_fn(output.float(), labels, loss_mask)

    return loss

class GPTModel(Module):
    r"""
    The Generative Pre-trained Transformer (GPT) is a decoder-only Transformer model.

    Args:
        config (TransformerConfig): The config of the transformer model. For details, please refer to TransformerConfig.
        num_tokentypes (int, optional): size of the token-type embeddings.
            If > 0, using tokentypes embedding. Default: ``0``.
        parallel_output (bool, optional): Specifies whether return paralleled output on
            each tensor parallel rank. Default: ``True``.
        pre_process (bool, optional): When using pipeline parallel, indicate whether it's the first stage. Default:
            ``True``.
        post_process (bool, optional): When using pipeline parallel, indicate whether it's the last stage. Default:
            ``True``.
        kwargs (dict): Other input.

    Inputs:
        - **tokens** (Tensor) - Input indices. Shape :math:`(B, S)`.
        - **position_ids** (Tensor) - Position offset. Shape :math:`(B, S)`.
        - **attention_mask** (Tensor) - Attention mask. Shape :math:`(B, S)`.
        - **loss_mask** (Tensor) - Loss mask. Shape :math:`(B, S)`.
        - **retriever_input_ids** (Tensor, optional) - Retriever input token indices. Shape: Depends on the input shape
          of the retrieval task. Default: ``None``.
        - **retriever_position_ids** (Tensor, optional) - Retriever input position indices. Shape: Depends on the input
          shape of the retrieval task. Default: ``None``.
        - **labels** (Tensor, optional) - Tensor of shape :math:`(N, )`. The ground truth label of the sample.
          Default: ``None``.
        - **tokentype_ids** (Tensor, optional) - List of token type ids to be fed to a model.
          Shape :math:`(B, S)`. Default: ``None``.
        - **inference_params** (Tensor, optional) - Inference parameters. Used to specify specific settings during
          inference, such as maximum generation length, max batch size, etc. Default: ``None``.

    Outputs:
        - Returns gpt loss or hidden states.

    Supported Platforms:
        ``Ascend``

    Examples:
        .. note::
            Before running the following examples, you need to configure the environment variables.

            For Ascend devices, it is recommended to use the msrun startup method without any third-party
            or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/docs/en/master/model_train/parallel/msrun_launcher.html>`_
            for more details.

        >>> import os
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication.management import init
        >>> from mindspeed_ms.core.parallel_state import initialize_model_parallel
        >>> from mindspeed_ms.core.config import (
        ...     ModelParallelConfig,
        ...     TrainingConfig,
        ...     DatasetConfig,
        ...     TransformerConfig
        ... )
        >>> from mindspeed_ms.legacy.model.gpt_model import GPTModel
        >>> ms.set_context(device_target='Ascend', mode=ms.PYNATIVE_MODE)
        >>> init()
        >>> initialize_model_parallel()
        >>> os.environ['HCCL_BUFFSIZE'] = "200"
        >>> batch_size = 8
        >>> seq_length = 32
        >>> parallel_config = ModelParallelConfig()
        >>> data_config = DatasetConfig(batch_size=batch_size)
        >>> training_config = TrainingConfig(parallel_config=parallel_config)
        >>> config = TransformerConfig(vocab_size=128,
        ...                            seq_length=seq_length,
        ...                            num_layers=4,
        ...                            num_attention_heads=4,
        ...                            num_query_groups=32,
        ...                            hidden_size=64,
        ...                            ffn_hidden_size=256,
        ...                            parallel_config=parallel_config,
        ...                            training_config=training_config,
        ...                            dataset_config=data_config)
        >>> gpt_model = GPTModel(config)
        >>> input_data = Tensor(np.random.random((batch_size, seq_length)).astype(np.int32))
        >>> attention_mask = Tensor(np.zeros((batch_size, 1, seq_length, seq_length)).astype(np.int32))
        >>> loss_mask = Tensor(np.random.random((batch_size, seq_length)).astype(np.int32))
        >>> lm_output = gpt_model(tokens=input_data,
        ...                       position_ids=None,
        ...                       attention_mask=attention_mask,
        ...                       loss_mask=loss_mask)
        >>> print(lm_output.shape)
        (32, 8, 128)
    """
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 **kwargs):

        super().__init__(config)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = config.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = config.fp16_lm_cross_entropy
        self.seq_length = config.seq_length
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.dataset_config.batch_size

        self.eod = kwargs['eod'] if 'eod' in kwargs else None
        self.reset_position_ids = kwargs['reset_position_ids'] if 'reset_position_ids' in kwargs else False
        self.cp_size = get_context_parallel_world_size()
        self.cp_rank = get_context_parallel_rank()
        if self.eod:
            self.eod_mask = EosMask(self.batch_size, config.seq_length, self.eod, self.reset_position_ids)


        self.language_model, _ = get_language_model(config=config,
                                                    encoder_attn_mask_type=AttnMaskType.causal,
                                                    num_tokentypes=num_tokentypes,
                                                    pre_process=self.pre_process,
                                                    post_process=self.post_process,
                                                    add_pooler=False)

        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                       bias=False,
                                                       compute_dtype=config.compute_dtype)
            self.loss = LossWithMask(VocabParallelCrossEntropy())

        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """
        set input_tensor to model

        Args:
            input_tensor (Tensor): the input tensor.
        """
        self.language_model.set_input_tensor(input_tensor)

    def construct(self, tokens, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """GPT model construct"""

        if (position_ids is None or attention_mask is None) and self.eod:
            position_ids, attention_mask = self.eod_mask(tokens)

        if self.cp_size > 1:
            seq_dim = 1
            tokens = ops.chunk(tokens, self.cp_size, seq_dim)[self.cp_rank]
            position_ids = ops.chunk(position_ids, self.cp_size, seq_dim)[self.cp_rank]
            loss_mask = ops.chunk(loss_mask, self.cp_size, seq_dim)[self.cp_rank]
            if labels is not None:
                labels = ops.chunk(labels, self.cp_size, seq_dim)[self.cp_rank]

        lm_output = self.language_model(tokens,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        retriever_input_ids=retriever_input_ids,
                                        retriever_position_ids=retriever_position_ids,
                                        retriever_attn_mask=retriever_attn_mask,
                                        inference_params=inference_params)

        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            return post_language_model_processing(self.parallel_lm_logits,
                                                  self.loss,
                                                  lm_output,
                                                  labels,
                                                  logit_weights,
                                                  self.parallel_output,
                                                  self.fp16_lm_cross_entropy,
                                                  loss_mask)

        return lm_output
