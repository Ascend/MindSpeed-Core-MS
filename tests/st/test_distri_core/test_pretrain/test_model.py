# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test model"""
import numpy as np
import mindspore as ms

from mindspeed_ms.legacy.model.language_model import get_language_model
from mindspeed_ms.legacy.model.module import Module
from mindspeed_ms.legacy.model import ParallelLMLogits
from mindspeed_ms.core.tensor_parallel import VocabParallelCrossEntropy
from mindspeed_ms.training import get_args


class TestModel(Module):
    """ model to test """
    # pylint: disable=W0613
    def __init__(
            self,
            config,
            num_tokentypes=0,
            add_pooler=False,
            encoder_attn_mask_type=None,
            pre_process=True,
            post_process=True,
            **kwargs,
    ):
        args = get_args()
        super().__init__(config, share_embeddings_and_output_weights=args.untie_embeddings_and_output_weights)

        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.language_model, _ = get_language_model(config=config,
                                                    encoder_attn_mask_type=encoder_attn_mask_type,
                                                    num_tokentypes=num_tokentypes,
                                                    pre_process=pre_process,
                                                    post_process=post_process,
                                                    add_pooler=add_pooler)
        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=config,
                                                       bias=False,
                                                       compute_dtype=config.compute_dtype)
            self.loss_func = VocabParallelCrossEntropy()
        if not self.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        self.language_model.set_input_tensor(input_tensor)

    def construct(
            self,
            tokens,
            attention_mask,
            position_ids=None,
            labels=None,
            tokentype_ids=None,
            inference_params=None,
            loss_mask=None,
    ):
        """forward"""
        batch_size, seq_length = tokens.shape
        if position_ids is None:
            position_ids = ms.Tensor(np.broadcast_to(np.expand_dims(np.arange(seq_length), 0),
                                                     (batch_size, seq_length)),
                                     dtype=tokens.dtype)
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
                                                       True,
                                                       self.fp16_lm_cross_entropy)
        return lm_output

    # pylint: disable=W0613
    def post_language_model_processing(
            self,
            lm_output,
            labels,
            logit_weights,
            loss_mask,
            parallel_output,
            fp16_lm_cross_entropy,
    ):
        """post process"""
        logits = self.parallel_lm_logits(lm_output, logit_weights, parallel_output)

        if labels is None:
            return logits.swapaxes(0, 1).contiguous()
        labels = labels.swapaxes(0, 1).contiguous()

        if fp16_lm_cross_entropy:
            loss = self.loss_func(logits, labels)
        else:
            loss = self.loss_func(logits.float(), labels)

        loss = loss.swapaxes(0, 1).contiguous()

        return loss
