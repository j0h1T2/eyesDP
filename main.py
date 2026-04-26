import torch
import os
from torch.utils.data import ConcatDataset, DataLoader, random_split
from core.model import EventEyeTrackerModel
from core.dataset import get_real_dataloaders
from tools.train import train_model
from core.dataset import EVEyeDataset
# 引入新 SNN 模型
#from core.snn_model import SpikingEyeTracker

def main():
    print("=== 超高频眼动事件检测：真实数据训练开始 ===")
    
    # 请确保这两个文件名和你下载的文件名完全一致！
    LEFT_EVENTS = os.path.join('data', 'train', 'user1_s1_left_events.txt')
    RIGHT_EVENTS = os.path.join('data', 'train', 'user1_s1_right_events.txt')
    LABELS_FILE = os.path.join('data', 'train', 'user1_s1_labels.txt')
    
    # 基础配置
    IN_CHANNELS = 5       # 现在有5个通道：X, Y, T, P, 距离
    CNN_OUT_CHANNELS = 16
    LSTM_HIDDEN = 32
    NUM_CLASSES = 2       # 现在只有 0(注视) 和 1(扫视) 两类！
    BATCH_SIZE = 64
    EPOCHS = 30
    SEQ_LEN = 200        # 时间步长
    
    # 2. 分别创建左眼和右眼的数据集对象
    # 注意：两个数据集都会自动应用你写在 dataset.py 里的 0.4s 时间补偿
    print("加载左眼数据...")
    left_dataset = EVEyeDataset(LEFT_EVENTS, LABELS_FILE, seq_len=SEQ_LEN)
    
    print("加载右眼数据...")
    right_dataset = EVEyeDataset(RIGHT_EVENTS, LABELS_FILE, seq_len=SEQ_LEN)
    
    # 3. 使用 ConcatDataset 合并数据集
    full_dataset = ConcatDataset([left_dataset, right_dataset])
    print(f"✅ 数据合并完成！总样本数: {len(full_dataset)}")
    
    # 4. 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # 5. 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = EventEyeTrackerModel(
        input_channels=IN_CHANNELS, 
        cnn_out_channels=CNN_OUT_CHANNELS, 
        #hidden_channels=16,
        lstm_hidden=LSTM_HIDDEN, 
        num_classes=NUM_CLASSES
    )

    # 3. 开始训练
    print("\n--- 开始模型训练 ---")
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=0.001)

if __name__ == "__main__":
    main()
