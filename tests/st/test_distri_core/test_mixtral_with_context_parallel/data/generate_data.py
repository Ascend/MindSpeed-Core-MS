"""Generate data"""
import numpy as np

# 加载 .npy 文件
# data = np.load('golden_mixtral_input_and_loss.npy', allow_pickle=True)

new_data = {'input': np.random.randint(0, 100, (2, 17)),  # 生成一个形状为 (2, 17) 的随机整数数组
            'loss': np.random.rand(10)}

# 打印数据
print(new_data)

np.save('./golden_mixtral_input_and_loss_cp.npy', new_data)
