from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class UnpadSoftMaxOpBuilder(AtbOpBuilder):
    OP_NAME = "unpad_softmax"

    def __init__(self):
        super(UnpadSoftMaxOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/unpad_softmax.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']