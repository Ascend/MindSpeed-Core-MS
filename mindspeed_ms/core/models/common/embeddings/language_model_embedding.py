# Copyright 2024 Huawei Technologies Co., Ltd
""" embedding """

from typing import Literal

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindspeed_ms.core import tensor_parallel
from mindspeed_ms.core.transformer import Module
from mindspeed_ms.core.transformer import TransformerConfig
from mindspeed_ms.core.tensor_parallel import get_rng_tracer


class LanguageModelEmbedding(Module):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head . Defaults to 0.
    """

    def __init__(
            self,
            config: TransformerConfig,
            vocab_size: int,
            max_sequence_length: int,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            num_tokentypes: int = 0,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.config.sequence_parallel
        )

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
        )

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = nn.Embedding(self.max_sequence_length,
                                                    self.config.hidden_size,
                                                    embedding_table=self.config.init_method,
                                                    dtype=ms.int32)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = nn.Embedding(self.max_sequence_length,
                                                     self.config.hidden_size,
                                                     embedding_table=self.config.init_method,
                                                     dtype=ms.int32)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = mint.nn.Dropout(self.config.hidden_dropout)

        # init comm op
        self.scatter_to_sequence_parallel_region = tensor_parallel.ScatterToSequenceParallelRegion(
            need_to_swapaxes=False
        )

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        ops.assign(self.word_embeddings, ops.zeros_like(self.word_embeddings))
        self.word_embeddings.weight.shared = True
        if self.use_position_embedding:
            ops.assign(self.position_embeddings, ops.zeros_like(self.position_embeddings))
            self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            ops.assign(self.tokentype_embeddings, ops.zeros_like(self.tokentype_embeddings))
            self.tokentype_embeddings.weight.shared = True

    def construct(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)
        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.swapaxes(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings:
                embeddings = self.scatter_to_sequence_parallel_region(embeddings)
            if self.config.clone_scatter_output_in_embedding:
                raise NotImplementedError("`clone_scatter_output_in_embedding` is not supported for now.")
            with get_rng_tracer().rng_fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
