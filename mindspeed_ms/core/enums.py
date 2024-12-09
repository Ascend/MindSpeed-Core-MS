"""Enum Classes."""

import enum


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
    retro_encoder = 3
    retro_decoder = 4
    retro_decoder_with_retriever = 5


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2
    retro_encoder = 3
    retro_decoder = 4
