import sys
sys.path.append('.')
import numpy as np
from core.dataset import calculate_ivt_labels
from scipy.signal import medfilt

print("Testing median filter in calculate_ivt_labels...")

# 创建一些测试数据
np.random.seed(42)
n_samples = 100
timestamps = np.arange(n_samples) * 0.01  # 10ms间隔
coords = np.random.randn(n_samples, 2) * 0.1 + np.array([0.5, 0.5])  # 随机坐标

print(f"Generated {n_samples} timestamps and coordinates")

# 调用函数
labels = calculate_ivt_labels(timestamps, coords, velocity_threshold=0.15)

print(f"Labels shape: {labels.shape}")
print(f"Labels dtype: {labels.dtype}")
print(f"Unique values: {np.unique(labels)}")

# 检查是否为0和1的整型数组
assert labels.dtype == np.int64 or labels.dtype == int, f"Expected int dtype, got {labels.dtype}"
assert set(np.unique(labels)).issubset({0, 1}), f"Labels should be 0 or 1, got {np.unique(labels)}"

# 手动验证中值滤波效果
# 首先计算未滤波的原始标签
velocities = [0.0]
for i in range(1, len(coords)):
    dt = timestamps[i] - timestamps[i-1]
    dist = np.linalg.norm(coords[i] - coords[i-1])
    velocities.append(dist / dt if dt > 1e-6 else 0.0)
raw_labels = (np.array(velocities) > 0.15).astype(int)

# 应用中值滤波
if len(raw_labels) >= 7:
    filtered_manual = medfilt(raw_labels, kernel_size=7).astype(int)
else:
    filtered_manual = raw_labels

# 比较函数输出与手动计算
assert np.array_equal(labels, filtered_manual), "Function output doesn't match manual calculation"

print("\nTesting edge cases...")
# 测试数据少于7个的情况
small_timestamps = np.arange(5) * 0.01
small_coords = np.random.randn(5, 2) * 0.1
small_labels = calculate_ivt_labels(small_timestamps, small_coords)
print(f"Small dataset (n=5) labels: {small_labels}")
assert len(small_labels) == 5

# 测试数据长度为1的情况
single_timestamps = np.array([0.0])
single_coords = np.array([[0.5, 0.5]])
single_labels = calculate_ivt_labels(single_timestamps, single_coords)
print(f"Single point labels: {single_labels}")
assert len(single_labels) == 1
assert single_labels[0] == 0

print("\n✅ All tests passed! Median filter is correctly applied.")
print(f"Raw labels stats: 0s={np.sum(raw_labels==0)}, 1s={np.sum(raw_labels==1)}")
print(f"Filtered labels stats: 0s={np.sum(labels==0)}, 1s={np.sum(labels==1)}")

# 展示一个示例：原始标签和滤波后标签的区别
print("\nExample of filtering effect (first 20 points):")
print("Raw labels:   ", raw_labels[:20])
print("Filtered labels:", labels[:20])