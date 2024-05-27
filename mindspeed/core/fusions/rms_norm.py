import os
from functools import wraps
import torch
import torch_npu
from torch import nn
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.training import get_args


def rms_norm_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        self.use_fused_rmsnorm = args.use_fused_rmsnorm
    return wrapper


def rms_norm_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, x):
        if int(os.getenv('NPU_ASD_ENABLE', '0')):
            from torch_npu.utils import register_asd_hook
            register_asd_hook(x, self.weight)
        if self.use_fused_rmsnorm:
            return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
        return fn(self, x)
    return wrapper
