"""GPT Model."""

import mindspore.common.dtype as mstype
from mindspore import ops, nn
from mindspeed_ms.training.global_vars import get_args
from mindspeed_ms.core.tensor_parallel import VocabParallelCrossEntropy
from mindspeed_ms.legacy.model.enums import AttnMaskType
from .module import Module
from .language_model import get_language_model
from .language_model import ParallelLMLogits


class PostLanguageModelProcessing(nn.Cell):
    """Post Language Model Processing."""

    def __init__(self,
                 config,
                 **kwargs):
        super().__init__(config, **kwargs)
        self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                   bias=False,
                                                   compute_dtype=config.compute_dtype)
        self.loss_func = VocabParallelCrossEntropy()
        self.config = config

    def construct(self,
                  lm_output,
                  labels,
                  logit_weights,
                  loss_mask,
                  parallel_output,
                  fp16_lm_cross_entropy):
        """ Construct function of Post Language Model Processing. """
        logits = self.parallel_lm_logits(lm_output, logit_weights, parallel_output)

        if labels is None:
            return logits.swapaxes(0, 1).contiguous()

        logits = logits.swapaxes(0, 1).contiguous()

        if fp16_lm_cross_entropy:
            assert logits.dtype == mstype.float16
            loss = self.loss_func(logits, labels)
        else:
            loss = self.loss_func(logits.float(), labels)

        loss_mask = loss_mask.float()

        loss = ops.sum(loss * loss_mask) / loss_mask.sum()
        return loss


class GPTModel(Module):
    """GPT Model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 **kwargs):
        args = get_args()
        super().__init__(config, **kwargs)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.set_model_key()
        self.language_model, _ = get_language_model(config=config,
                                                    encoder_attn_mask_type=AttnMaskType.causal,
                                                    num_tokentypes=num_tokentypes,
                                                    pre_process=self.pre_process,
                                                    post_process=self.post_process,
                                                    add_pooler=False)
        self.post_language_model_processing = PostLanguageModelProcessing(config)
        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def set_model_key(self):
        """ set model key for differentiate PipelineCell process """
        self.model_key = "gpt_model"

    def construct(self, tokens, position_ids, attention_mask,
                  labels=None, tokentype_ids=None, inference_params=None, loss_mask=None):
        """ Construct function of GPT Model. """
        lm_output = self.language_model(tokens,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        inference_params=inference_params)

        if self.post_process:
            if self.untie_embeddings_and_output_weights:
                logit_weights = self.language_model.output_layer.weight
            else:
                logit_weights = self.shared_embedding_or_output_weight()
            return self.post_language_model_processing(lm_output,
                                                       labels,
                                                       logit_weights,
                                                       loss_mask,
                                                       self.parallel_output,
                                                       self.fp16_lm_cross_entropy)
        return lm_output
