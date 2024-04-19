import torch
import torch_npu
from torch.library import impl
from ascendspeed.op_builder.npu_mm_all_reduce_add_rms_norm_builder import MatmulAllReduceAddRmsNormOpBuilder
from ascendspeed.op_builder.builder import AS_LIBRARY

ascendspeed_ops = MatmulAllReduceAddRmsNormOpBuilder().load()


@impl(AS_LIBRARY, "npu_mm_all_reduce_add_rms_norm", "PrivateUse1")
def npu_mm_all_reduce_add_rms_norm_single(x1,
                                          x2,
                                          residual,
                                          gamma,
                                          hcom,
                                          reduce_op='sum',
                                          epsilon=1e-06,
                                          bias=None,
                                          antiquant_scale=None,
                                          antiquant_offset=None,
                                          dequant_scale=None,
                                          antiquant_group_size=0,
                                          comm_turn=0):
    y, normOut = ascendspeed_ops.npu_mm_all_reduce_add_rms_norm(x1,
                                                                x2,
                                                                residual,
                                                                gamma,
                                                                hcom,
                                                                reduce_op,
                                                                epsilon,
                                                                bias,
                                                                antiquant_scale,
                                                                antiquant_offset,
                                                                dequant_scale,
                                                                antiquant_group_size,
                                                                comm_turn)
    return (y.view(residual.shape), normOut.view(residual.shape))


def npu_mm_all_reduce_add_rms_norm(*args, **kwargs):
    return torch.ops.ascendspeed.npu_mm_all_reduce_add_rms_norm(*args, **kwargs)