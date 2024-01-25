import os
import sys
import types
from functools import wraps
import torch
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


def multi_tensor_l2norm(overflow_buf, tensor_lists, per_parameter):
    total_norm = 0.0
    norm_type = 2.0
    ret_per_tensor = [] if per_parameter else None
    for grads_for_norm in tensor_lists:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm ** norm_type
        if per_parameter:
            ret_per_tensor.append(total_norm.clone())
    if not tensor_lists:
        grad_norm = torch.cuda.FloatTensor([0])
        total_norm = grad_norm ** norm_type
    return total_norm ** (1 / norm_type), ret_per_tensor


def multi_tensor_scale(overflow_buf, tensor_lists, scale):
    if len(tensor_lists) != 2:
        raise AssertionError('The size of tensor list must be 2, but got {}'.format(len(tensor_lists)))
    if len(tensor_lists[0]) != len(tensor_lists[1]):
        raise AssertionError('The size of tensor list must be same, but got {} and {}'.format(len(tensor_lists[0]),
                                                                                              len(tensor_lists[1])))

    with torch.no_grad():
        for i in range(len(tensor_lists[0])):
            tensor_lists[1][i].copy_(tensor_lists[0][i] * scale)


def exe_adaptation():
    # Need replace modules before import megatron
    sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
    setattr(sys.modules['transformer_engine'], 'pytorch', torch.nn)
    setattr(sys.modules['transformer_engine'].pytorch, 'LayerNormLinear', torch.nn.Module)
    setattr(sys.modules['transformer_engine'].pytorch, 'DotProductAttention', torch.nn.Module)
    sys.modules['fused_layer_norm_cuda'] = types.ModuleType('fused_layer_norm_cuda')
    sys.modules['amp_C'] = types.ModuleType('amp_C')
    setattr(sys.modules['amp_C'], 'multi_tensor_l2norm', multi_tensor_l2norm)
    setattr(sys.modules['amp_C'], 'multi_tensor_scale', multi_tensor_scale)
    sys.modules['flash_attn.flash_attn_interface'] = types.ModuleType('flash_attn_flash_attn_interface')
    setattr(sys.modules['flash_attn.flash_attn_interface'], 'flash_attn_unpadded_func', torch.nn.Module)

    # Torch and Apex monkey patching
    apex.optimizers.FusedAdam = torch.optim.AdamW  # replace apex fused adam
    apex.multi_tensor_apply.multi_tensor_applier = multi_tensor_applier
    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)  # replace npu type to gpu type
    torch.Tensor.type = type_wrapper(torch.Tensor.type)  # replace npu type to gpu type
    torch.Tensor.view = ensure_contiguous(torch.Tensor.view)  # patch view to ensure tensor is contiguous

    # Megatron core monkey patching
    import megatron.core.tensor_parallel
    import megatron.core.pipeline_parallel
    from .arguments import parse_args_decorator, validate_args_decorator
    from .core.pipeline_parallel.p2p_communication import _batched_p2p_ops
    from .core.tensor_parallel.random import _set_cuda_rng_state
    from .core.tensor_parallel.layers import VocabParallelEmbeddingForward
    from .core.tensor_parallel.cross_entropy import _VocabParallelCrossEntropyForward
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .core.fusions.rms_norm import rms_norm_init, rms_norm_forward
    from .core.fusions.transformer import parallel_mlp_init, flash_self_attention_forward
    from .core.fusions.rotary_pos_embedding import apply_fused_rotary_pos_emb

    megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops = _batched_p2p_ops  # send recv bug
    megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state  # default_generators need replace after set_device
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = VocabParallelEmbeddingForward
    megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = _VocabParallelCrossEntropyForward
    megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction
    megatron.core.fusions.fused_layer_norm.FastLayerNormFN = FastLayerNormFN
    megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax
    megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax
    megatron.core.fusions.fused_softmax.ScaledSoftmax = ScaledSoftmax
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
    megatron.core.transformer.attention.apply_rotary_pos_emb = apply_fused_rotary_pos_emb

    apex.normalization.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine

    # Megatron others modules monkey patching
    try:
        import megatron.initialize
        from .initialize import _compile_dependencies, set_jit_fusion_options

        megatron.initialize._compile_dependencies = _compile_dependencies  # remove cuda kernel compile
        megatron.initialize.set_jit_fusion_options = set_jit_fusion_options  # remove cuda jit nvfuser
        megatron.model.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction
        megatron.model.fused_layer_norm.FastLayerNormFN = FastLayerNormFN
        megatron.model.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine

        megatron.model.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax
        megatron.model.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax
        megatron.model.fused_softmax.ScaledSoftmax = ScaledSoftmax
        megatron.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
        megatron.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
        megatron.model.rms_norm.RMSNorm.__init__ = rms_norm_init
        megatron.model.rms_norm.RMSNorm.forward = rms_norm_forward
        megatron.model.transformer.ParallelMLP.__init__ = parallel_mlp_init
        megatron.model.transformer.FlashSelfAttention.forward = flash_self_attention_forward
        megatron.model.transformer.apply_rotary_pos_emb = apply_fused_rotary_pos_emb

        # Megatron wrappers
        megatron.initialize.parse_args = parse_args_decorator(megatron.initialize.parse_args)
        megatron.arguments.parse_args = parse_args_decorator(megatron.arguments.parse_args)
        megatron.initialize.validate_args = validate_args_decorator(megatron.initialize.validate_args)

        if int(os.getenv('NPU_DETECT', '0')):
            from .core.tensor_parallel.layers import embedding_wrapper, linear_wrapper
            from .core.fusions.fused_layer_norm import layernorm_wrapper
            from .training import train_step

            megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__ = embedding_wrapper(
                megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__)
            megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__ = linear_wrapper(
                megatron.core.tensor_parallel.layers.ColumnParallelLinear.__init__)
            megatron.core.tensor_parallel.layers.RowParallelLinear.__init__ = linear_wrapper(
                megatron.core.tensor_parallel.layers.RowParallelLinear.__init__)
            megatron.model.rms_norm.RMSNorm.__init__ = layernorm_wrapper(
                megatron.model.rms_norm.RMSNorm.__init__)
            megatron.model.RMSNorm.__init__ = layernorm_wrapper(
                megatron.model.RMSNorm.__init__)
            megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__ = layernorm_wrapper(
                megatron.model.fused_layer_norm.MixedFusedLayerNorm.__init__)

            import megatron.training
            megatron.training.train_step = train_step
    except ModuleNotFoundError:
        pass

    # accelerate package will check TE on sys.modules，so we need remove this patch
    del sys.modules['transformer_engine']


exe_adaptation()
