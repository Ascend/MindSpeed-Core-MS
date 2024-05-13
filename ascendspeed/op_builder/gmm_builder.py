import torch
from ascendspeed.op_builder.builder import AscendSpeedOpBuilder


class GMMOpBuilder(AscendSpeedOpBuilder):
    OP_NAME = "grouped_matmul"
    TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[:2])

    def __init__(self):
        super(GMMOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/gmm.cpp']

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
        if self.TORCH_MAJOR >= 2 and self.TORCH_MINOR >= 1:
            cpp_std = " -std=c++17"
        else:
            cpp_std = " -std=c++14"
        args.append(cpp_std)
        return args
