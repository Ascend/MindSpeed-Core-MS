# Copyright (c) 2024, Huawei Technologies.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional
import torch
from torch.library import Library, impl
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge_concrete_graph.ge_graph import Tensor, TensorSpec
from torchair.ge_concrete_graph.ge_graph import get_default_ge_graph, next_unique_name
from torchair.ge_concrete_graph.ge_graph import compat_as_bytes
from torchair.ge_concrete_graph.ge_graph import get_invalid_desc
from mindspeed.op_builder.builder import MindSpeedOpBuilder, AS_LIBRARY


class MatmulAllReduceAddRmsNormOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_mm_all_reduce_add_rms_norm"
    OP_PROTO = "npu_mm_all_reduce_add_rms_norm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, str hcom, *, \
        str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None, Tensor? antiquant_scale=None, Tensor? \
        antiquant_offset=None, Tensor? dequant_scale=None, int antiquant_group_size=0, int comm_turn=0) \
        -> (Tensor, Tensor)"

    def __init__(self):
        super(MatmulAllReduceAddRmsNormOpBuilder, self).__init__(self.OP_NAME)
        self.register_op_proto(self.OP_PROTO)
        self.register_op_ir()

    def sources(self):
        return ['ops/csrc/cann/npu_mm_all_reduce_add_rms_norm.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args
    
    def register_op_ir(self):
        @impl(AS_LIBRARY, "npu_mm_all_reduce_add_rms_norm", "Meta")
        def npu_mm_all_reduce_add_rms_norm_forward(x1, x2, residual, gamma, hcom, reduce_op='sum', epsilon=1e-6,
                                                   bias=None, antiquant_scale=None, antiquant_offset=None,
                                                   dequant_scale=None, antiquant_group_size=0, comm_turn=0):
            return (torch.empty_like(residual, dtype=residual.dtype),
                    torch.empty_like(residual, dtype=residual.dtype))
        
        @register_fx_node_ge_converter(torch.ops.mindspeed.npu_mm_all_reduce_add_rms_norm.default)
        def convert_npu_mm_all_reduce_add_rms_norm(
                x1: Tensor,
                x2: Tensor,
                residual: Tensor,
                gamma: Tensor,
                hcom: str,
                *,
                reduce_op: str = 'sum',
                epsilon: float = 1e-6,
                bias: Optional[Tensor] = None,
                antiquant_scale: Optional[Tensor] = None,
                antiquant_offset: Optional[Tensor] = None,
                dequant_scale: Optional[Tensor] = None,
                antiquant_group_size: int = 0,
                comm_turn: int = 0,
                meta_outputs: List[TensorSpec] = None
        ):
            # transpose_x1 is set to False by default
            transpose_x1 = False
            transpose_x2 = False
            '''"npu_mm_all_reduce_add_rms_norm(Tensor x1, Tensor x2, Tensor residual, Tensor gamma, str hcom,
                    *, str reduce_op='sum', float epsilon=1e-06, Tensor? bias=None, Tensor? antiquant_scale=None,
                    Tensor? antiquant_offset=None, Tensor? dequant_scale=None, int antiquant_group_size=0,
                    int comm_turn=0) -> (Tensor, Tensor)"'''
            return MatmulAllReduceAddRmsNorm(x1,
                                             x2,
                                             bias=bias,
                                             residual=residual,
                                             gamma=gamma,
                                             antiquant_scale=antiquant_scale,
                                             antiquant_offset=antiquant_offset,
                                             dequant_scale=dequant_scale,
                                             group=hcom,
                                             reduce_op=reduce_op,
                                             is_trans_a=transpose_x1,
                                             is_trans_b=transpose_x2,
                                             comm_turn=comm_turn,
                                             antiquant_group_size=antiquant_group_size,
                                             epsilon=epsilon)


def MatmulAllReduceAddRmsNorm(x1: Tensor,
                              x2: Tensor,
                              bias: Optional[Tensor],
                              residual: Tensor,
                              gamma: Tensor,
                              antiquant_scale: Optional[Tensor],
                              antiquant_offset: Optional[Tensor],
                              dequant_scale: Optional[Tensor],
                              *,
                              group: str,
                              reduce_op: str = "sum",
                              is_trans_a: bool = False,
                              is_trans_b: bool = False,
                              comm_turn: int = 0,
                              antiquant_group_size: int = 0,
                              epsilon: float = 0.000001,
                              dependencies=None,
                              node_name=None):
    """REG_OP(MatmulAllReduceAddRmsNorm)\n
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT8, DT_INT8, DT_INT4, DT_INT4}))\n
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .INPUT(residual, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .OPTIONAL_INPUT(dequant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .OUTPUT(norm_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_FLOAT16, DT_BF16, DT_FLOAT16, DT_BF16}))\n
    .REQUIRED_ATTR(group, String)\n
    .ATTR(reduce_op, String, "sum")\n
    .ATTR(is_trans_a, Bool, false)\n
    .ATTR(is_trans_b, Bool, false)\n
    .ATTR(comm_turn, Int, 0)\n
    .ATTR(antiquant_group_size, Int, 0)\n
    .ATTR(epsilon, Float, 1e-6)\n
    .OP_END_FACTORY_REG(MatmulAllReduceAddRmsNorm)
    """

    op = get_default_ge_graph().op.add()
    op.type = "MatmulAllReduceAddRmsNorm"
    op.name = next_unique_name(node_name, "MatmulAllReduceAddRmsNorm")

    # process dependices
    if dependencies is not None:
        for dependency in dependencies:
            op.input.append(dependency.controller)

    # process inputs
    op.input.append(x1.tensor)
    op.input_desc.add().CopyFrom(x1.desc)
    op.input_desc[-1].name = "x1"
    op.input.append(x2.tensor)
    op.input_desc.add().CopyFrom(x2.desc)
    op.input_desc[-1].name = "x2"
    if bias is not None:
        op.input.append(bias.tensor)
        op.input_desc.add().CopyFrom(bias.desc)
        op.input_desc[-1].name = "bias"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "bias"
    op.input.append(residual.tensor)
    op.input_desc.add().CopyFrom(residual.desc)
    op.input_desc[-1].name = "residual"
    op.input.append(gamma.tensor)
    op.input_desc.add().CopyFrom(gamma.desc)
    op.input_desc[-1].name = "gamma"
    if antiquant_scale is not None:
        op.input.append(antiquant_scale.tensor)
        op.input_desc.add().CopyFrom(antiquant_scale.desc)
        op.input_desc[-1].name = "antiquant_scale"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_scale"
    if antiquant_offset is not None:
        op.input.append(antiquant_offset.tensor)
        op.input_desc.add().CopyFrom(antiquant_offset.desc)
        op.input_desc[-1].name = "antiquant_offset"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "antiquant_offset"
    if dequant_scale is not None:
        op.input.append(dequant_scale.tensor)
        op.input_desc.add().CopyFrom(dequant_scale.desc)
        op.input_desc[-1].name = "dequant_scale"
    else:
        op.input.append('')
        op.input_desc.add().CopyFrom(get_invalid_desc())
        op.input_desc[-1].name = "dequant_scale"

    # process attrs
    op.attr["group"].s = compat_as_bytes(group)
    op.attr["reduce_op"].s = compat_as_bytes(reduce_op)
    op.attr["is_trans_a"].b = is_trans_a
    op.attr["is_trans_b"].b = is_trans_b
    op.attr["comm_turn"].i = comm_turn
    op.attr["antiquant_group_size"].i = antiquant_group_size
    op.attr["epsilon"].f = epsilon

    # process outputs
    output_index = 0
    op.output_desc.add().name = "y"
    y = Tensor(op, output_index)
    output_index += 1
    op.output_desc.add().name = "norm_out"
    norm_out = Tensor(op, output_index)
    output_index += 1

    # return outputs
    return y, norm_out