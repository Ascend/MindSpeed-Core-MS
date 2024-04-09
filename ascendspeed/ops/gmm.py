import torch
from ascendspeed.op_builder import GMMOpBuilder


class GMMFunction(torch.autograd.Function):
    ascendspeed_ops = GMMOpBuilder().load()

    @staticmethod
    def forward(ctx, x, weight, bias, group_list, group_type):
        if bias is not None and bias.requires_grad:
            raise ValueError("Bias is not supported for backward!")
        bias = [] if bias is None else [bias]
        outputs = GMMFunction.ascendspeed_ops.npu_gmm([x], [weight], bias, group_list, group_type)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        return outputs[0]

    @staticmethod
    def backward(ctx, grad_outputs):
        x, weight = ctx.saved_tensors
        dx, dw, dbias = GMMFunction.ascendspeed_ops.npu_gmm_backward(
            [grad_outputs], [x], [weight], ctx.group_list)
        dbias = None if len(dbias) == 0 else dbias[0]
        
        return dx[0], dw[0], dbias, None, None


def npu_gmm(x, weight, *, bias=None, group_list=None, group_type=-1):
    return GMMFunction.apply(x, weight, bias, group_list, group_type)
