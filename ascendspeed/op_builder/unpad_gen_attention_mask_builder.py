from ascendspeed.op_builder.atb_builder import AtbOpBuilder

class UnpadGenAttentionMaskOpBuilder(AtbOpBuilder):
    OP_NAME = "unpad_gen_attention_mask"

    def __init__(self):
        super(UnpadGenAttentionMaskOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/atb/unpad_gen_attention_mask.cpp',
                'ops/csrc/atb/utils/atb_adapter.cpp',
                'ops/csrc/atb/utils/adapter.cpp']