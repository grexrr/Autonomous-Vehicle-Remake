import numpy as np

# 原始数据
ref_trajectory = np.array([
    [0.0, 0.0, 0.0, 1],      # 第0个点（起点）
    [1.0, 0.5, 0.1, 1],      # 第1个点
    [2.0, 1.0, 0.2, 1],      # 第2个点
    [3.0, 1.5, 0.3, 1],      # 第3个点
    [4.0, 2.0, 0.4, 1],      # 第4个点
])

print("原始数据:")
for i, point in enumerate(ref_trajectory[1:], 1):
    print(f"索引{i}: {point}")