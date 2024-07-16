import torch
import torch_npu
from torch import Tensor
from functools import wraps
from megatron.training import get_args
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding


def apply_fused_rotary_pos_emb_bshd_wrapper(fn):
    @wraps(fn)
    def wrapper(t: Tensor, freqs: Tensor, rotary_interleaved: bool = False) -> Tensor:
        _args = get_args()
        if _args.use_fused_rotary_pos_emb:
            rot_dim = freqs.shape[-1]
            t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
            cos_ = torch.cos(freqs).to(t.dtype)
            sin_ = torch.sin(freqs).to(t.dtype)
            t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_).to(t.dtype)
            return torch.cat((t, t_pass), dim=-1)
        return fn(t, freqs, rotary_interleaved)

    return wrapper


def rotary_embedding_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        if _args.rotary_base:
            kwargs["rotary_base"] = _args.rotary_base
        fn(self, *args, **kwargs)

    return wrapper
