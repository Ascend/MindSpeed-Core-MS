import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadRopeOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)

class TestNPUUnpadRope(DistributedTest):
    world_size = 1

    def supported_op_exec(self, q, k, cos, sin,seqlen):
        ntoken = q.shape[0]
        batch = seqlen.shape[0]
        hidden_size = q.shape[1]
        head_size = cos.shape[1]
        head_num = hidden_size // head_size
        q_list = []
        k_list = []
        offset = 0
        for i, _ in enumerate(range(batch)):
            cur_seqlen = seqlen[i]
            next_offset = offset + cur_seqlen
            qlayer = q[offset:next_offset].view(cur_seqlen, head_num, head_size)
            klayer = k[offset:next_offset].view(cur_seqlen, head_num, head_size)
            curcos = cos[:cur_seqlen].unsqueeze(1)
            cursin = sin[:cur_seqlen].unsqueeze(1)
            qlayer = (qlayer * curcos) + (rotate_half(qlayer) * cursin)
            klayer = (klayer * curcos) + (rotate_half(klayer) * cursin)
            outq = qlayer.view(cur_seqlen, hidden_size)
            q_list.append(outq)
            outk = klayer.view(cur_seqlen, hidden_size)
            k_list.append(outk)
            offset = next_offset
        q_sum = torch.cat(tuple(q_list), dim=0)
        k_sum = torch.cat(tuple(k_list), dim=0)
        return [q_sum, k_sum]


    def custom_op_exec(self, q, k, cos, sin, seqlen, rotaryCoeff, cosFormat):
        ascendspeed_ops = UnpadRopeOpBuilder().load()
        q_embed, k_embed = ascendspeed_ops.npu_unpad_rope(q, k, cos, sin, seqlen, rotaryCoeff, cosFormat)
        return q_embed, k_embed

    def golden_compare(self, out_tensors, golden_out_tensors):
        return (torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
                and torch.allclose(out_tensors[1].float(), golden_out_tensors[1].float(), rtol=0.001, atol=0.001))

    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_rope(self):
        batch = 4
        rotaryCoeff = 2
        cosFormat = 1
        headDim = 128
        maxseqlen = 2048

        seqlen = np.random.randint(1, maxseqlen, size=batch, dtype=np.int32)
        hiddensizeQ = 1024
        hiddensizeK = 1024
        q = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeQ)).astype(np.float16)
        k = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeK)).astype(np.float16)
        cos = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
        sin = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
        q = torch.tensor(q).npu().half().contiguous()
        k = torch.tensor(k).npu().half().contiguous()
        cos = torch.tensor(cos).npu().half().contiguous()
        sin = torch.tensor(sin).npu().half().contiguous()
        seqlen = torch.tensor(seqlen).npu().int().contiguous()
        q_embed = torch.zeros_like(q,device=q.device).half().contiguous()
        k_embed = torch.zeros_like(q,device=q.device).half().contiguous()

        golden_out_tensors = self.supported_op_exec(q, k, cos, sin,seqlen)
        q_embed, k_embed = self.custom_op_exec(q, k, cos, sin, seqlen, rotaryCoeff, cosFormat)
        out_tensors = [q_embed, k_embed]
        assert self.golden_compare(out_tensors, golden_out_tensors)
