import pytest
import torch
import torch_npu
import numpy as np

from ascendspeed.op_builder import UnpadRopeOpBuilder
from unit_tests.common import DistributedTest

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]

class TestNPUUnpadRope(DistributedTest):
    world_size = 1

    def supported_op_exec(self, qembedgrad, kembedgrad, cos, sin, seqlen):
        cos_list=[cos[:x, :] for x in seqlen]
        sin_list=[sin[:x, :] for x in seqlen]
        cos = torch.cat(cos_list, dim=0)
        sin = torch.cat(sin_list, dim=0)
        sin1=sin[:,:64]
        sin2=sin[:,64:]
        rohqgsin=torch.cat((sin2,-sin1),dim=-1)

        q_grad=torch.zeros_like(qembedgrad)
        bs=int(qembedgrad.shape[1]/128)
        for i in range(bs):
            q_grad[:,i*128:(i+1)*128] = qembedgrad[:,i*128:(i+1)*128] *(cos+rohqgsin)

        k_grad=torch.zeros_like(kembedgrad)
        for i in range(bs):
            k_grad[:,i*128:(i+1)*128] = kembedgrad[:,i*128:(i+1)*128] *(cos+rohqgsin)
        return [q_grad,k_grad]


    def custom_op_exec(self, qembedgrad, kembedgrad, cos, sin,seqlen):
        ascendspeed_ops = UnpadRopeOpBuilder().load()
        q_grad, k_grad = ascendspeed_ops.npu_unpad_rope_grad(qembedgrad, kembedgrad, cos, sin, seqlen)
        return q_grad, k_grad

    def golden_compare(self, out_tensors, golden_out_tensors):
        return (torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
            and torch.allclose(out_tensors[1].float(), golden_out_tensors[1].float(), rtol=0.001, atol=0.001))


    @pytest.mark.skipif(DEVICE_NAME != 'Ascend910B', reason='device type is not supported, skip this UT!')
    def test_npu_unpad_rope_grad(self):
        batch = 16
        headDim = 128
        maxseqlen=200

        seqlen =np.random.randint(1, maxseqlen, size=batch, dtype=np.int32)
        hiddensizeQ = 2048
        hiddensizeK = 2048
        qembedgrad = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeQ)).astype(np.float16)
        kembedgrad = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeK)).astype(np.float16)
        cos = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
        sin = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
        qembedgrad=torch.tensor(qembedgrad).npu().half()
        kembedgrad=torch.tensor(kembedgrad).npu().half()
        cos=torch.tensor(cos).npu().half()
        sin=torch.tensor(sin).npu().half()

        golden_out_tensors = self.supported_op_exec(qembedgrad, kembedgrad, cos, sin,seqlen)
        q_grad, k_grad = self.custom_op_exec(qembedgrad, kembedgrad, cos, sin, seqlen)
        out_tensors = [q_grad, k_grad]
        assert self.golden_compare(out_tensors, golden_out_tensors)
