"""This is a temp mock class for adapting MindSpeed op_builder import"""

def npu_gmm():
    raise NotImplementedError
setattr(npu_gmm, "Tensor", None)

def npu_gmm_v2():
    raise NotImplementedError
setattr(npu_gmm_v2, "Tensor", None)
