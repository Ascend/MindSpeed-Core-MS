import torch
import torch_npu
from functools import wraps
from megatron.training import get_args
from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half


def apply_fused_rotary_pos_emb_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.use_fused_rotary_pos_emb:
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)
        fn(self, *args, **kwargs)

    return wrapper


def rotary_embedding_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.rotary_base:
            kwargs["rotary_base"] = _args.rotary_base
        fn(self, *args, **kwargs)

    return wrapper
