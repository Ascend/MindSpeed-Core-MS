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
"""Language Module."""

from mindspore import Tensor, ops
import mindspore.communication.comm_func as comm_func
import mindspore.communication._comm_helper as _comm_helper
from mindspeed_ms.tools import logger

from mindspeed_ms.core import parallel_state
from mindspeed_ms.core.tensor_parallel import VocabParallelCrossEntropy
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.transformer.transformer_config import TransformerConfig


# pylint: disable=R1705,W0613,W0212
class LanguageModule(Module):
    """Base language module that has common helper functions used across GPT, BERT etc.

    Args:
        config (TransformerConfig): Input transformer config for the model
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        self.vocab_parallel_cross_entropy = VocabParallelCrossEntropy()

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.swapaxes(0, 1).contiguous()
        if self.config.cross_entropy_loss_fusion:
            raise NotImplementedError("cross_entropy_loss_fusion not supported for now.")
        loss = self.vocab_parallel_cross_entropy(logits.float(), labels)

        # [s b] => [b, s]
        loss = loss.swapaxes(0, 1).contiguous()
        return loss

    def shared_embedding_or_output_weight(self):
        """ get share weight in first stage and last stage """

        if self.pre_process:
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def setup_embeddings_and_output_layer(self) -> None:
        """Sets up embedding layer in first stage and output layer in last stage.

        This function initializes word embeddings in the final stage when we are
        using pipeline parallelism and sharing word embeddings, and sets up param
        attributes on the embedding and output layers.
        """

        # Set `is_embedding_or_output_parameter` attribute.
        if self.pre_process:
            self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
        if self.post_process and self.output_layer.weight is not None:
            self.output_layer.weight.is_embedding_or_output_parameter = True

        if not self.share_embeddings_and_output_weights:
            return

        if self.pre_process and not self.post_process:
            assert parallel_state.is_pipeline_first_stage()
            self.shared_embedding_or_output_weight().shared_embedding = True

        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            ops.assign(self.output_layer.weight, ops.zeros_like(self.output_layer.weight))
            self.output_layer.weight.shared = True
            self.output_layer.weight.shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.

        # Ensure that first and last stages have the same initial parameter
        # values.
        if _comm_helper._is_initialized():
            if parallel_state.is_rank_in_embedding_group():
                weight = self.shared_embedding_or_output_weight()
                shared_weight_value = comm_func.all_reduce(weight.value(),
                                                           group=parallel_state.get_embedding_group())[0]
                ops.assign(weight, shared_weight_value)

        elif not getattr(LanguageModule, "embedding_warning_printed", False):
            logger.warning(
                "Distributed processes aren't initialized, so the output layer "
                "is not initialized with weights from the word embeddings. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )
            LanguageModule.embedding_warning_printed = True
