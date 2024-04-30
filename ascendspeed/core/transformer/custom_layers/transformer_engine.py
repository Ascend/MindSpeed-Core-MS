import torch
import torch_npu
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model.rms_norm import RMSNorm

        
class PTNorm:
    """
    A conditional wrapper to initialize an instance of Transformer-Engine's
    `LayerNorm` or `RMSNorm` based on input
    """

    # TODO should we ditch normalization config and just use spec to choose LayerNorm vs RMSNorm?
    def __new__(
        cls, config: TransformerConfig, hidden_size: int, eps: float = 1e-5,
    ):
        if config.normalization == "LayerNorm":
            instance = nn.LayerNorm(
                normalized_shape=hidden_size,
                eps=eps,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(
                dim=hidden_size,
                eps=eps,
                sequence_parallel=config.sequence_parallel,
                use_fused_rmsnorm=True,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
