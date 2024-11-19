"""Test data"""
import numpy as np

# 加载 .npy 文件
data = np.load('golden_mixtral_input_and_loss_cp.npy', allow_pickle=True)

# 打印数据
print(data)

data = data.tolist()
print(data['input'].shape)
