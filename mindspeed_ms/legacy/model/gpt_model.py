
"""GPT model."""

import mindspore.common.dtype as mstype
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.legacy.model.enums import AttnMaskType
from mindspeed_ms.legacy.model.language_model import get_language_model
from mindspeed_ms.legacy.model.transformer import ParallelLMLogits
from mindspeed_ms.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from mindspeed_ms.training.loss_func import LossWithMask
from mindspeed_ms.legacy.model.eos_mask import EosMask

def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy, loss_mask):
    """ gpt model post process forward """
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        # [s b h] -> [b s h]
        return output.swapaxes(0, 1)

    # [s b] -> [b s]
    output = output.swapaxes(0, 1)
    loss_mask = loss_mask.reshape(-1)

    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels, loss_mask)
    else:
        loss = loss_fn(output.astype(mstype.float32), labels, loss_mask)

    return loss

class GPTModel(Module):
    """
    GPT model
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
        self.reset_position_ids = kwargs['reset_position_ids']
        if self.eod:
            self.eod_mask = EosMask(self.batch_size, config.seq_length, self.eod, self.reset_position_id)

        self.set_model_key()

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
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def set_model_key(self):
        """ set model key for differentiate PipelineCell process """
        self.model_key = "gpt_model"

    def construct(self, tokens, position_ids, attention_mask, loss_mask,
                  retriever_input_ids=None,
                  retriever_position_ids=None,
                  retriever_attn_mask=None,
                  labels=None, tokentype_ids=None, inference_params=None):
        """GPT model construct"""

        if (position_ids is None or attention_mask is None) and self.eod:
            position_ids, attention_mask = self.eod_mask(tokens)

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
