from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class PadSeqLenOpBuilder(AtbOpBuilder):
    OP_NAME = "pad_seqlen"

    def __init__(self):
        super(PadSeqLenOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/pad_seqlen.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']