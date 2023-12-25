import sys
import types
from functools import wraps
import torch
import torch_npu
import apex
from torch_npu.contrib import transfer_to_npu


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs).replace('npu', 'cuda')

    return wrapper


# Patch view method to ensure tensor is contiguous before performing view
def ensure_contiguous(fn):
    def wrapper(tensor, *args, **kwargs):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return fn(tensor, *args, **kwargs)

    return wrapper


def multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    return op(noop_flag_buffer, tensor_lists, *args)


def exe_adaptation():
    # Need replace modules before import megatron
    sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
    setattr(sys.modules['transformer_engine'], 'pytorch', torch.nn)
    setattr(sys.modules['transformer_engine'].pytorch, 'LayerNormLinear', torch.nn.Module)
    setattr(sys.modules['transformer_engine'].pytorch, 'DotProductAttention', torch.nn.Module)
    sys.modules['fused_layer_norm_cuda'] = types.ModuleType('fused_layer_norm_cuda')
    sys.modules['amp_C'] = types.ModuleType('amp_C')

    # About torch replace
    from .optimizer.adam import AdamW
    apex.optimizers.FusedAdam = AdamW  # replace apex fused adam
    apex.multi_tensor_apply.multi_tensor_applier = multi_tensor_applier
    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)  # replace npu type to gpu type
    torch.Tensor.type = type_wrapper(torch.Tensor.type)  # replace npu type to gpu type
    torch.Tensor.view = ensure_contiguous(torch.Tensor.view)  # patch view to ensure tensor is contiguous

    import megatron
    import megatron.optimizer
    import megatron.core.pipeline_parallel
    from .initialize import _compile_dependencies, set_jit_fusion_options
    from .core.pipeline_parallel.p2p_communication import _batched_p2p_ops
    from .core.tensor_parallel.random import _set_cuda_rng_state
    from .optimizer.clip_grads import clip_grad_norm_fp32
    from .model.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN
    from .model.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .arguments import parse_args_decorator

    # Megatron monkey patching
    megatron.initialize._compile_dependencies = _compile_dependencies  # remove cuda kernel compile
    megatron.initialize.set_jit_fusion_options = set_jit_fusion_options  # remove cuda jit nvfuser

    megatron.model.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction
    megatron.model.fused_layer_norm.FastLayerNormFN = FastLayerNormFN
    megatron.model.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax
    megatron.model.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax
    megatron.model.fused_softmax.ScaledSoftmax = ScaledSoftmax

    megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
    megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops = _batched_p2p_ops  # send recv bug
    megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state  # default_generators need replace after set_device
    megatron.initialize.parse_args = parse_args_decorator(megatron.initialize.parse_args)

    for k, v in sys.modules.items():
        if 'megatron' in k and hasattr(v, 'clip_grad_norm_fp32'):
            setattr(v, 'clip_grad_norm_fp32', clip_grad_norm_fp32)  # replace  clip_grad_norm_fp32


exe_adaptation()
