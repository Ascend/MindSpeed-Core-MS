import os
from functools import wraps

import torch


class FusedLayerNormAffineFunction:
    @staticmethod
    def apply(input_, weight, bias, normalized_shape, eps):
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


class FastLayerNormFN:
    @staticmethod
    def apply(input_, weight, bias, eps):
        normalized_shape = torch.Size(weight.numel())
        return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


def fused_layer_norm_affine(input_, weight, bias, normalized_shape, eps):
    return torch.nn.functional.layer_norm(input_, normalized_shape, weight, bias, eps)


def layernorm_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        res = fn(self, *args, **kwargs)
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.hook_module.hook_module import HOOKModule

            def vocabparallelembedding_hook(grad):
                HOOKModule.layernorm_list.append(torch.norm(grad))

            self.weight.register_hook(vocabparallelembedding_hook)
        return res

    return wrapper
