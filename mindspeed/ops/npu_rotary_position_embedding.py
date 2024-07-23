import torch
from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder
rope_ops = RotaryPositionEmbeddingOpBuilder().load()


def npu_rotary_position_embedding(x, cos, sin, mode=0):
    return rope_ops.npu_rotary_position_embedding(x, cos, sin, mode)
