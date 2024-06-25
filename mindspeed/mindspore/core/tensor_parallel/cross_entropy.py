# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import mindtorch.torch as torch
from  mindspeed.mindspore.mindspore_adaptor import bprop_commn

@staticmethod
def vocab_parallel_cross_entropy_backward(ctx, grad_output):
    # Retreive tensors from the forward path.
    softmax, target_mask, masked_target_1d = ctx.saved_tensors
    label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

    # All the inputs have softmax as thier gradient.
    grad_input = softmax
    # For simplicity, work with the 2D gradient.
    partition_vocab_size = softmax.size()[-1]
    grad_2d = grad_input.view(-1, partition_vocab_size)

    # Add the gradient from matching classes.
    arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

    softmax_update = 1.0 - target_mask.view(-1).float()

    if label_smoothing > 0:
        voc = vocab_size - 1
        smoothing = 1.0
        if voc != 0:
            smoothing = label_smoothing * vocab_size / voc
        grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
        average_grad = 1 / 32000
        if vocab_size != 0:
            average_grad = 1 / vocab_size
        grad_2d[arange_1d, :] -= smoothing * average_grad
    else:
        grad_2d[arange_1d, masked_target_1d] -= softmax_update

    grad_input = grad_2d.view_as(grad_input)
    # Finally elementwise multiplication with the output gradients.
    grad_input.mul_(grad_output.unsqueeze(dim=-1))

    return grad_input, None, None


def bocab_parallel_cross_entropy_bprop(self, vocab_parallel_logits, target, label_smoothing, output, grad_output):
    return bprop_commn(self, grad_output)