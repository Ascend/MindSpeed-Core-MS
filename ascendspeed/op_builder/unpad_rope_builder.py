from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class UnpadRopeOpBuilder(AtbOpBuilder):
    OP_NAME = "unpad_rope"

    def __init__(self):
        super(UnpadRopeOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/unpad_rope.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']