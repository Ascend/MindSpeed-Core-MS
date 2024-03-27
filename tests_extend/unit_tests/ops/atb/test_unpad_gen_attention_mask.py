import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadGenAttentionMaskOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadGenAttentionMask(DistributedTest):
    world_size = 1

    def supported_op_exec(self, in_tensors, seqlen, headNum):
        out = []
        for i, s in enumerate(seqlen):
            for _ in range(headNum):
                out.append(in_tensors[i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def custom_op_exec(self, in_tensors, seqlen, headNum):
        ascendspeed_ops = UnpadGenAttentionMaskOpBuilder().load()
        return ascendspeed_ops.npu_unpad_gen_attention_mask(in_tensors, seqlen, headNum)

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_gen_attention_mask(self):
        batch = 4
        maxseqlen = 2048
        seq_len = torch.randint(100, 300, [batch,]).to(torch.int32)
        seqlen = np.array(seq_len)
        headNum = 8
        in_tensors = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
        out = []
        shapeOut = sum(map(lambda x: x**2, seqlen)) * headNum
        for _ in range(shapeOut):
            out.append(0.1)

        golden_out_tensors = self.supported_op_exec(in_tensors, seqlen, headNum)
        result = self.custom_op_exec(in_tensors, seqlen, headNum)
        res = result.cpu().numpy().tolist()
        out_tensors = [result]
        assert torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
