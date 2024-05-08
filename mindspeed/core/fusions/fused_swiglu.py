import torch_npu


def fused_swiglu(x):
    return torch_npu.npu_swiglu(x, dim=-1)