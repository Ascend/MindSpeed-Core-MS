from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class UnpadStridedBatchMatmulOpBuilder(AtbOpBuilder):
    OP_NAME = "unpad_strided_batch_matmul"

    def __init__(self):
        super(UnpadStridedBatchMatmulOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/unpad_strided_batch_matmul.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']