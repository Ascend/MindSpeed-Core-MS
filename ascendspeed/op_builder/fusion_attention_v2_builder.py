from ascendspeed.op_builder.builder import AscendSpeedOpBuilder


class FusionAttentionV2OpBuilder(AscendSpeedOpBuilder):
    OP_NAME = "fusion_attention_v2"

    def __init__(self):
        super(FusionAttentionV2OpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/fusion_attention_v2.cpp']

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
