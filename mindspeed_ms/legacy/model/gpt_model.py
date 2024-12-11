"""GPT Model."""

import mindspore.common.dtype as mstype
from mindspeed_ms.training.global_vars import get_args, get_tokenizer
from mindspeed_ms.core.tensor_parallel import VocabParallelCrossEntropy
from mindspeed_ms.legacy.model.enums import AttnMaskType
from mindspeed_ms.legacy.model.eos_mask import EosMask
from .module import Module
from .language_model import get_language_model
from .language_model import ParallelLMLogits


def post_language_model_processing(parallel_lm_logits, loss_fn, lm_output, labels, logit_weights,
                                   parallel_output, fp16_lm_cross_entropy):
    """ gpt model post process forward """
    output = parallel_lm_logits(lm_output, logit_weights, parallel_output)

    if labels is None:
        # [s, b, h] -> [b, s, h]
        return output.swapaxes(0, 1).contiguous()

    # [b, s] -> [s, b]
    labels = labels.swapaxes(0, 1).contiguous()

    if fp16_lm_cross_entropy:
        if output.dtype != mstype.float16:
            raise ValueError(f"When fp16_lm_cross_entropy=True, output should be float16, but got {output.dtype}")
        loss = loss_fn(output, labels)
    else:
        loss = loss_fn(output.float(), labels)

    # [s, b] -> [b, s]
    loss = loss.swapaxes(0, 1).contiguous()

    return loss


class GPTModel(Module):
    """GPT Model."""
    # pylint: disable=W0613
    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 **kwargs):
        args = get_args()
        super().__init__(config, share_embeddings_and_output_weights=args.untie_embeddings_and_output_weights)
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.eod = get_tokenizer().eod if args.reset_attention_mask else None
        if self.eod:
            self.eod_mask = EosMask(args.batch_size,
                                    args.seq_length,
                                    self.eod,
                                    args.reset_position_ids)

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
            self.loss_func = VocabParallelCrossEntropy()

        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """ set input_tensor to model """
        self.language_model.set_input_tensor(input_tensor)

    def construct(self, tokens, position_ids, attention_mask,
                  labels=None, tokentype_ids=None, inference_params=None):
        """ Construct function of GPT Model. """
        if (position_ids is None or attention_mask is None) and self.eod:
            position_ids, attention_mask = self.eod_mask(tokens)

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
            return post_language_model_processing(self.parallel_lm_logits,
                                                  self.loss_func,
                                                  lm_output,
                                                  labels,
                                                  logit_weights,
                                                  self.parallel_output,
                                                  self.fp16_lm_cross_entropy)
        return lm_output
