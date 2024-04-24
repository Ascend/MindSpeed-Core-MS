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
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

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


def repeat_interleave(inputs, repeats, dim):
    shape = inputs.shape
    new_shape = shape[:dim + 1] + (repeats,) + shape[dim + 1:]
    out_shape = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
    return inputs.unsqueeze(dim + 1).expand(new_shape).reshape(out_shape)


def mcore_adaptation(aspm):
    from ascendspeed.core.transformer.transformer_block import DummyTENorm, final_layernorm_wrapper
    from ascendspeed.core.transformer.dot_product_attention import dot_product_attention_forward_wrapper
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    aspm.register_patch('megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec', get_gpt_layer_local_spec)
    aspm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TENorm', DummyTENorm)
    aspm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers', final_layernorm_wrapper)
    aspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward', dot_product_attention_forward_wrapper)


def moe_adaptation(aspm):
    from .model.transformer import SwitchMLP_forward_wrapper, SwitchMLP_init_wrapper
    from .tokenizer import build_tokenizer_decorator
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.__init__', SwitchMLP_init_wrapper)
    aspm.register_patch('megatron.legacy.model.transformer.SwitchMLP.forward', SwitchMLP_forward_wrapper)
    aspm.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer_decorator)


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
    from .optimizer.adamw import AdamW
    apex.optimizers.FusedAdam = AdamW  # replace apex fused adam
    apex.multi_tensor_apply.multi_tensor_applier = multi_tensor_applier
    torch.nn.parameter.Parameter.type = type_wrapper(torch.nn.parameter.Parameter.type)  # replace npu type to gpu type
    torch.Tensor.type = type_wrapper(torch.Tensor.type)  # replace npu type to gpu type
    torch.Tensor.view = ensure_contiguous(torch.Tensor.view)  # patch view to ensure tensor is contiguous
    torch.Tensor.repeat_interleave = repeat_interleave # replace npu implementation of torch.repeat_interleave

    # Megatron core monkey patching
    import megatron.core.tensor_parallel
    import megatron.core.pipeline_parallel
    from .arguments import parse_args_decorator, validate_args_decorator, core_transformer_config_from_args_wrapper
    from .core.pipeline_parallel.p2p_communication import _batched_p2p_ops
    from .core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
    from .core.pipeline_parallel.schedules import get_tensor_shapes
    from .core.tensor_parallel.random import _set_cuda_rng_state, backward
    from .core.tensor_parallel.layers import VocabParallelEmbeddingForward
    from .core.tensor_parallel.cross_entropy import _VocabParallelCrossEntropyForward
    from .core.fusions.fused_layer_norm import FusedLayerNormAffineFunction, FastLayerNormFN, fused_layer_norm_affine
    from .core.fusions.fused_softmax import is_kernel_available, ScaledUpperTriangMaskedSoftmax, ScaledMaskedSoftmax, \
        ScaledSoftmax, forward_fused_softmax
    from .core.fusions.rms_norm import rms_norm_init, rms_norm_forward
    from .model.transformer import parallel_mlp_init, flash_self_attention_forward
    from .core.fusions.rotary_pos_embedding import apply_fused_rotary_pos_emb
    from .core.fusions.rotary_pos_embedding import RotaryEmbedding_wrapper
    from .model.transformer import core_attention_wrapper, core_attention_forward, ParallelAttention_wrapper, ParallelAttention_forward_wrapper

    megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops = _batched_p2p_ops  # send recv bug
    megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving = forward_backward_pipelining_with_interleaving # context parallel bug
    megatron.core.pipeline_parallel.schedules.get_tensor_shapes = get_tensor_shapes # context parallel bug
    megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state  # default_generators need replace after set_device
    megatron.core.tensor_parallel.random.CheckpointFunction.backward = backward
    megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward = VocabParallelEmbeddingForward
    megatron.core.tensor_parallel.cross_entropy._VocabParallelCrossEntropy.forward = _VocabParallelCrossEntropyForward
    megatron.core.fusions.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction
    megatron.core.fusions.fused_layer_norm.FastLayerNormFN = FastLayerNormFN
    megatron.core.fusions.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax
    megatron.core.fusions.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax
    megatron.core.fusions.fused_softmax.ScaledSoftmax = ScaledSoftmax
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
    megatron.core.models.common.embeddings.apply_rotary_pos_emb_bshd = apply_fused_rotary_pos_emb
    megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__ = RotaryEmbedding_wrapper(
        megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__
    )

    apex.normalization.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine


    import megatron.training.initialize
    import megatron.training.training
    from .training import pretrain
    from .initialize import _compile_dependencies, set_jit_fusion_options, coc_registration_wrapper, mc2_wrapper
    from .optimizer.optimizer import mixed_precision_optimizer_step, fp32_optimizer_step, reuse_fp32_param_init_wrapper, optimizer_config_init_wrapper
    from .core.tensor_parallel.layers import row_parallel_nocomm_optimizer_wrapper
    from .core.transformer.transformer import parallel_transformer_layer_forward_wrapper, parallel_transformer_checkpointed_forward_wrapper
    from .core.transformer.attention import attention_wrapper
    from .utils import get_batch_on_this_cp_rank

    megatron.training.initialize._compile_dependencies = _compile_dependencies  # remove cuda kernel compile
    megatron.training.utils.get_batch_on_this_cp_rank = get_batch_on_this_cp_rank
    megatron.legacy.model.fused_layer_norm.FusedLayerNormAffineFunction = FusedLayerNormAffineFunction
    megatron.legacy.model.fused_layer_norm.FastLayerNormFN = FastLayerNormFN
    megatron.legacy.model.fused_layer_norm.fused_layer_norm_affine = fused_layer_norm_affine

    megatron.legacy.model.fused_softmax.ScaledUpperTriangMaskedSoftmax = ScaledUpperTriangMaskedSoftmax
    megatron.legacy.model.fused_softmax.ScaledMaskedSoftmax = ScaledMaskedSoftmax
    megatron.legacy.model.fused_softmax.ScaledSoftmax = ScaledSoftmax
    megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.is_kernel_available = is_kernel_available  # replace kernel check
    megatron.legacy.model.fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = forward_fused_softmax
    megatron.legacy.model.rms_norm.RMSNorm.__init__ = rms_norm_init
    megatron.legacy.model.rms_norm.RMSNorm.forward = rms_norm_forward
    megatron.legacy.model.transformer.ParallelMLP.__init__ = parallel_mlp_init
    megatron.legacy.model.transformer.FlashSelfAttention.forward = flash_self_attention_forward
    megatron.legacy.model.transformer.ParallelAttention.__init__ = ParallelAttention_wrapper(
        megatron.legacy.model.transformer.ParallelAttention.__init__)
    megatron.legacy.model.transformer.ParallelAttention.forward = ParallelAttention_forward_wrapper(
        megatron.legacy.model.transformer.ParallelAttention.forward)
    megatron.legacy.model.transformer.CoreAttention.__init__ = core_attention_wrapper(
        megatron.legacy.model.transformer.CoreAttention.__init__)
    megatron.legacy.model.transformer.CoreAttention.forward = core_attention_forward
    megatron.core.transformer.attention.Attention.__init__ = attention_wrapper(
        megatron.core.transformer.attention.Attention.__init__)

    # Megatron wrappers
    megatron.training.initialize.parse_args = parse_args_decorator(megatron.training.initialize.parse_args)
    megatron.training.arguments.parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    megatron.training.initialize.validate_args = validate_args_decorator(megatron.training.initialize.validate_args)
    megatron.training.arguments.core_transformer_config_from_args = core_transformer_config_from_args_wrapper(
        megatron.training.arguments.core_transformer_config_from_args)
    megatron.core.tensor_parallel.layers.RowParallelLinear.forward = row_parallel_nocomm_optimizer_wrapper(
        megatron.core.tensor_parallel.layers.RowParallelLinear.forward)
    megatron.legacy.model.transformer.ParallelTransformerLayer.forward = parallel_transformer_layer_forward_wrapper(
        megatron.legacy.model.transformer.ParallelTransformerLayer.forward
    )
    megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward = parallel_transformer_checkpointed_forward_wrapper(
        megatron.legacy.model.transformer.ParallelTransformer._checkpointed_forward
    )

    # MoE relative.
    from .core.parallel_state import initialize_model_parallel_decorator
    from .core.parallel_state import destroy_model_parallel_decorator
    from .core.parallel_state import get_expert_parallel_group
    from .core.parallel_state import get_expert_parallel_rank
    from .core.parallel_state import get_expert_model_parallel_rank
    from .core.parallel_state import get_expert_parallel_world_size
    from .core.parallel_state import get_expert_model_parallel_world_size
    from .core.parallel_state import set_expert_model_parallel_rank
    from .core.parallel_state import set_expert_model_parallel_world_size

    megatron.core.parallel_state.initialize_model_parallel = initialize_model_parallel_decorator(
        megatron.core.parallel_state.initialize_model_parallel)
    megatron.core.parallel_state.destroy_model_parallel = destroy_model_parallel_decorator(
        megatron.core.parallel_state.destroy_model_parallel)
    setattr(
        megatron.core.parallel_state, "get_expert_parallel_group", get_expert_parallel_group)
    setattr(
        megatron.core.parallel_state, "get_expert_parallel_rank", get_expert_parallel_rank)
    setattr(
        megatron.core.parallel_state, "get_expert_model_parallel_rank", get_expert_model_parallel_rank)
    setattr(
        megatron.core.parallel_state, "get_expert_parallel_world_size", get_expert_parallel_world_size)
    setattr(
        megatron.core.parallel_state, "get_expert_model_parallel_world_size", get_expert_model_parallel_world_size)
    setattr(
        megatron.core.parallel_state, "set_expert_model_parallel_rank", set_expert_model_parallel_rank)
    setattr(
        megatron.core.parallel_state, "set_expert_model_parallel_world_size", set_expert_model_parallel_world_size)
    megatron.core.mpu = megatron.core.parallel_state

    import megatron.core.optimizer
    megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step = mixed_precision_optimizer_step
    megatron.core.optimizer.optimizer.FP32Optimizer.step = fp32_optimizer_step
    megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__ = \
        reuse_fp32_param_init_wrapper(megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__)
    megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__ = \
        optimizer_config_init_wrapper(megatron.core.optimizer.optimizer_config.OptimizerConfig.__init__)

    if int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        from .core.memory.memory_fragmentation.pluggable_allocator_adpator import change_allocator
        change_allocator()

        import megatron.training
        from .core.memory.memory_fragmentation.memory_recorder import memory_recorder_wrap
        megatron.training.setup_model_and_optimizer = memory_recorder_wrap(
            megatron.training.setup_model_and_optimizer)

        from .core.memory.memory_fragmentation.malloc_recorder import malloc_recorder_wrap
        megatron.training.train_step = malloc_recorder_wrap(megatron.training.train_step)

        from .core.memory.memory_fragmentation.optimizer_init_precise import optimizer_init_wrap
        megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step = optimizer_init_wrap(
            megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step)
        import megatron.training
        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(megatron.model.transformer.ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_decorator
        megatron.training.setup_model_and_optimizer = setup_model_and_optimizer_decorator(
            megatron.training.setup_model_and_optimizer)

    if int(os.getenv('ADAPTIVE_RECOMPUTING', '0')) and not int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        from .core.memory.adaptive_recomputing.pluggable_allocator_adpator import change_allocator
        change_allocator()
        import megatron.training
        from .core.memory.adaptive_recomputing.adaptive_recompute import allowed_recomputing_module_wrapper
        allowed_recomputing_module_wrapper(megatron.model.transformer.ParallelTransformerLayer)
        from .core.memory.adaptive_recomputing.adaptive_recompute import setup_model_and_optimizer_decorator
        megatron.training.setup_model_and_optimizer = setup_model_and_optimizer_decorator(
            megatron.training.setup_model_and_optimizer)

    #new patch ways
    from ascendspeed.patch_utils import AscendSpeedPatchesManager as aspm
    aspm.register_patch('megatron.training.initialize.set_jit_fusion_options', set_jit_fusion_options)
    aspm.register_patch('megatron.training.training.pretrain', pretrain)
    if int(os.getenv('ASCEND_MC2', '0')):
        aspm.register_patch('megatron.training.initialize.initialize_megatron', mc2_wrapper)
    aspm.register_patch('megatron.training.initialize.initialize_megatron', coc_registration_wrapper)

    mcore_adaptation(aspm)
    moe_adaptation(aspm)

    aspm.apply_patches()

    if int(os.getenv('ADAPTIVE_RECOMPUTING', '0')) or int(os.getenv('MEMORY_FRAGMENTATION', '0')):
        megatron.training.initialize_megatron = megatron.training.initialize.initialize_megatron

    # accelerate package will check TE on sys.modules，so we need remove this patch
    del sys.modules['transformer_engine']


exe_adaptation()
