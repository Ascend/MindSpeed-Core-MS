from typing import List
import torch
from mindspeed.op_builder import GMMOpBuilder


class GMMFunction(torch.autograd.Function):
    mindspeed_ops = GMMOpBuilder().load()

    @staticmethod
    def forward(ctx, x, weight, bias, group_list, group_type):
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported to compute gradient!")
        if (x.requires_grad or weight.requires_grad) and group_type != 0:
            raise ValueError("group_type must be zero to compute gradients of x and weight!")
        bias = [] if bias is None else [bias]
        outputs = GMMFunction.mindspeed_ops.npu_gmm([x], [weight], bias, group_list, group_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        dx, dw, dbias = GMMFunction.mindspeed_ops.npu_gmm_backward(
            [grad_outputs], [x], [weight], ctx.group_list)
        dbias = None if len(dbias) == 0 else dbias[0]
        
        return dx[0], dw[0], dbias, None, None


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError("arg0 must be a torch.Tensor, got {}".format(type(x)))
    if not isinstance(weight, torch.Tensor):
        raise TypeError("arg1 must be a torch.Tensor, got {}".format(type(weight)))
    if not isinstance(bias, (torch.Tensor, type(None))):
        raise TypeError("bias must be a torch.Tensor or None, got {}".format(type(bias)))
    if not (isinstance(group_list, type(None)) \
        or (isinstance(group_list, list) and all(isinstance(x, int) for x in group_list))):
        raise TypeError("group_list must be a List of int or None, got {}".format(type(group_list)))
    if not isinstance(group_type, (int, type(None))):
        raise TypeError("group_type must be an int or None, got {}".format(type(group_type)))
    return GMMFunction.apply(x, weight, bias, group_list, group_type)
