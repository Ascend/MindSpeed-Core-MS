import torch
import torch_npu
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.legacy.model.rms_norm import RMSNorm

        
class PTNorm:
    """
    Conditional Initialization of Transformer-Engine’s LayerNorm or RMSNorm Instance
    """
    
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
            )
            instance.use_fused_rmsnorm = True
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance
