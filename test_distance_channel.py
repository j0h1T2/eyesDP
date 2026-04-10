import sys
sys.path.append('.')
import torch
from core.dataset import EVEyeDataset
import numpy as np

print("Testing distance channel addition...")

# 使用现有的数据文件路径
LEFT_EVENTS = 'data/train/user1_s1_left_events.txt'
LABELS_FILE = 'data/train/user1_s1_labels.txt'

try:
    # 创建数据集实例，使用 seq_len=200
    dataset = EVEyeDataset(LEFT_EVENTS, LABELS_FILE, seq_len=200)
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # 获取第一个样本
        feature_tensor, label_tensor = dataset[0]
        
        print(f"Feature tensor shape: {feature_tensor.shape}")
        print(f"Expected shape: (5, 200)")
        print(f"Label tensor: {label_tensor}")
        
        # 验证形状
        assert feature_tensor.shape == (5, 200), f"Expected (5, 200), got {feature_tensor.shape}"
        
        # 验证通道顺序
        print("\nChannel contents:")
        print(f"Channel 0 (X): mean={feature_tensor[0].mean().item():.6f}, std={feature_tensor[0].std().item():.6f}")
        print(f"Channel 1 (Y): mean={feature_tensor[1].mean().item():.6f}, std={feature_tensor[1].std().item():.6f}")
        print(f"Channel 2 (T): mean={feature_tensor[2].mean().item():.6f}, std={feature_tensor[2].std().item():.6f}")
        print(f"Channel 3 (P): mean={feature_tensor[3].mean().item():.6f}, std={feature_tensor[3].std().item():.6f}")
        print(f"Channel 4 (distance): mean={feature_tensor[4].mean().item():.6f}, std={feature_tensor[4].std().item():.6f}")
        
        # 验证距离通道的第一个元素应为0（因为diff后第一个距离为0）
        assert torch.allclose(feature_tensor[4, 0], torch.tensor(0.0)), f"First distance should be 0, got {feature_tensor[4, 0]}"
        
        # 验证X和Y是否已去中心化（第一个点应为0）
        assert torch.allclose(feature_tensor[0, 0], torch.tensor(0.0)), f"First X should be 0 after centering, got {feature_tensor[0, 0]}"
        assert torch.allclose(feature_tensor[1, 0], torch.tensor(0.0)), f"First Y should be 0 after centering, got {feature_tensor[1, 0]}"
        
        print("\n✅ Test passed! Distance channel added successfully.")
        
        # 额外检查几个样本
        print("\nChecking additional samples...")
        for i in range(min(5, len(dataset))):
            feature, label = dataset[i]
            assert feature.shape == (5, 200), f"Sample {i}: Expected (5, 200), got {feature.shape}"
            print(f"Sample {i}: shape OK, label={label}")
            
    else:
        print("⚠️ Dataset has no samples, but files exist.")
        
except FileNotFoundError as e:
    print(f"⚠️ File not found: {e}")
    print("Make sure data files exist at the expected paths.")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()