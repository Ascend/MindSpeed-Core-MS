import torch
import torch_npu
import numpy as np
from ascendspeed.op_builder import UnpadGenAttentionMaskOpBuilder

repeatTime=5

def golden_calc(in_tensors, seqlen, headNum):
    out = []
    for i, s in enumerate(seqlen):
        for _ in range(headNum):
            out.append(in_tensors[i, :, :s, :s].flatten())
    return [torch.hstack(out)]

def golden_compare(out_tensors, golden_out_tensors):
    return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

if __name__ == '__main__':
    for i in range(repeatTime):
        batch = 4
        maxseqlen = 2048
        seq_len = torch.randint(100, 300, [batch,]).to(torch.int32)
        seqlen = np.array(seq_len)
        headNum = 8
        a = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
        out = []
        shapeOut = sum(map(lambda x: x**2, seqlen)) * headNum
        for _ in range(shapeOut):
            out.append(0.1)

        ascendspeed_ops = UnpadGenAttentionMaskOpBuilder().load()
        result = ascendspeed_ops.npu_unpad_gen_attention_mask(a, seqlen, headNum)
        res = result.cpu().numpy().tolist()
        out_tensors = [result]
        golden_out_tensors = golden_calc(a, seqlen, headNum)
        res_compare = golden_compare(out_tensors, golden_out_tensors)
        print("res_compare:", res_compare)