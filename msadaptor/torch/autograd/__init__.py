"""autograd"""
from mindspore import Tensor as Variable
from .grad_mode import no_grad, enable_grad
from .function import value_and_grad, grad, Function, vjp, recompute_instance
from .variable import Variable

def _is_checkpoint_valid():
    return True