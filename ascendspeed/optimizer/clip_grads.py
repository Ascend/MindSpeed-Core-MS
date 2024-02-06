import os
import torch
from torch import inf
from apex.multi_tensor_apply import multi_tensor_applier
import amp_C


def clip_grad_norm_fp32(parameters, grads_for_norm,
                        max_norm, check_for_nan_in_grad,
                        norm_type=2, model_parallel_group=None):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            assert param.grad.type() == 'torch.cuda.FloatTensor'
            grads.append(param.grad.detach())

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.utils.silent_error import get_silent_check
            total_norm_cuda = torch.Tensor([total_norm_cuda, float(get_silent_check())]).npu()

        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=model_parallel_group)
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.utils.silent_error import set_silent_check
            set_silent_check(total_norm_cuda[1].item())
            total_norm_cuda = total_norm_cuda[0]

        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.cuda.IntTensor([0])
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False  # no per-parameter norm
                )
            else:
                grad_norm = torch.cuda.FloatTensor([0])
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Check individual rank grad norms are not NaN
        # prior to model-parallel all-reduce.
        if check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            assert not total_norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backwards pass. Device: {torch.cuda.current_device()}, '
                f'node: {os.uname()[1]}'
            )

        # Sum across all model-parallel GPUs.
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.utils.silent_error import get_silent_check
            total_norm = torch.Tensor([total_norm, float(get_silent_check())]).npu()
        torch.distributed.all_reduce(total_norm,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=model_parallel_group)
        if int(os.getenv('NPU_DETECT', '0')):
            from torch_npu.utils.silent_error import set_silent_check
            set_silent_check(total_norm[1].item())
            total_norm = total_norm[0]

        total_norm = total_norm.item() ** (1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.cuda.IntTensor([0])
        multi_tensor_applier(amp_C.multi_tensor_scale,
                             dummy_overflow_buf,
                             [grads, grads],
                             clip_coeff)

    return total_norm
