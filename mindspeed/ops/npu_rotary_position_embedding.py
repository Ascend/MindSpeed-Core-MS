import torch
from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder


class RotaryPositionEmbeddingFunction(torch.autograd.Function):
    mindspeed_ops = RotaryPositionEmbeddingOpBuilder().load()

    @staticmethod
    def forward(ctx, x, cos, sin, mode):
        output = RotaryPositionEmbeddingFunction.mindspeed_ops.\
            npu_rotary_position_embedding(x, cos, sin, mode)
        ctx.save_for_backward(x, cos, sin)
        ctx.mode = mode
        
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x, cos, sin = ctx.saved_tensors
        dx, dcos, dsin = RotaryPositionEmbeddingFunction.mindspeed_ops.\
            npu_rotary_position_embedding_backward(grad_outputs, cos, sin, x, ctx.mode)
        
        return dx, dcos, dsin, None
        

def npu_rotary_position_embedding(x, cos, sin, mode=0):
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input x must be a torch.Tensor, got {}".format(type(x)))
    if not isinstance(cos, torch.Tensor):
        raise TypeError("Input cos must be a torch.Tensor, got {}".format(type(cos)))
    if not isinstance(sin, torch.Tensor):
        raise TypeError("Input sin must be a torch.Tensor, got {}".format(type(sin)))
    if not isinstance(mode, (int, type(None))):
        raise TypeError("Input mode must be an int or None, got {}".format(type(mode)))
    return RotaryPositionEmbeddingFunction.apply(x, cos, sin, mode)
