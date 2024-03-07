from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class UnpadSeqLenOpBuilder(AtbOpBuilder):
    OP_NAME = "unpad_seqlen"

    def __init__(self):
        super(UnpadSeqLenOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/unpad_seqlen.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']