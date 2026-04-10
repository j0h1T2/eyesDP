import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
from scipy.signal import medfilt
from .preprocess import EventDataPreprocessor

def parse_tobii_gaze(gaze_file_path):
    """解析 Tobii 的 JSON 格式 gaze 数据（带防丢帧容错）"""
    timestamps = []
    gaze_coords = []
    
    with open(gaze_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if item.get('type') == 'gaze':
                    # 安全地获取 data 字典
                    data_dict = item.get('data', {})
                    # 只有当 gaze2d 存在且不为空时，才记录这个坐标
                    if 'gaze2d' in data_dict and data_dict['gaze2d'] is not None:
                        timestamps.append(item['timestamp'])
                        gaze_coords.append(data_dict['gaze2d'])
            except json.JSONDecodeError:
                continue
                
    return np.array(timestamps), np.array(gaze_coords)

def calculate_ivt_labels(timestamps, coords, velocity_threshold=0.20):
    """
    I-VT 算法：根据坐标变化计算角速度，生成 0(注视) 和 1(扫视) 标签
    velocity_threshold: 经验值，可能需要根据数据集调整
    """
    if len(coords) < 2:
        return np.zeros(len(coords), dtype=int)
        
    velocities = [0.0]
    for i in range(1, len(coords)):
        dt = timestamps[i] - timestamps[i-1]
        dist = np.linalg.norm(coords[i] - coords[i-1])
        # 简单处理：如果时间差极小，速度设为0
        velocities.append(dist / dt if dt > 1e-6 else 0.0)
        
    labels = (np.array(velocities) > velocity_threshold).astype(int)
    
    # 应用中值滤波去除孤立跳变（窗口大小7）
    if len(labels) >= 7:
        labels = medfilt(labels, kernel_size=7).astype(int)
    
    return labels

class EVEyeDataset(Dataset):
    def __init__(self, events_file, labels_file, seq_len=100):
        super(EVEyeDataset, self).__init__()
        self.seq_len = seq_len
        self.preprocessor = EventDataPreprocessor()
        
        print(f"正在读取事件文件: {events_file} ...")
        df_events = pd.read_csv(events_file, sep='\s+', header=None)
        raw_events = df_events.values
        
        self.events = np.zeros((len(raw_events), 4))
        self.events[:, 0] = raw_events[:, 1]  # X
        self.events[:, 1] = raw_events[:, 2]  # Y
        self.events[:, 2] = raw_events[:, 0]  # Timestamp
        self.events[:, 3] = raw_events[:, 3]  # Polarity

        # 我们把 X 和 Y 强行压缩到 0 ~ 1 之间
        self.events[:, 0] = self.events[:, 0] / 346.0
        self.events[:, 1] = self.events[:, 1] / 260.0
        # 极性 (Polarity) 原本是 0 或 1，也可以将其转为 -1 和 1 增强特征
        self.events[:, 3] = (self.events[:, 3] * 2) - 1.0


        print(f"读取到 {len(self.events)} 条事件数据。")
        
        print(f"正在读取标签文件: {labels_file} ...")
        tobii_timestamps, tobii_coords = parse_tobii_gaze(labels_file)
        self.raw_labels = calculate_ivt_labels(tobii_timestamps, tobii_coords)
        print(f"生成了 {len(self.raw_labels)} 个注视/扫视标签。")
        
        # ==========================================
        # 👑 核心提分点：统一时间轴（极其重要！）
        # ==========================================
        # 1. 将事件的微秒级时间戳减去起始时间，并转换为秒
        start_time_micro = self.events[0, 2]
        self.events[:, 2] = (self.events[:, 2] - start_time_micro) / 1000000.0
        
        # 2. 将 Tobii 的时间戳也减去它的起始时间，从 0 秒开始
        start_time_tobii = tobii_timestamps[0]
        tobii_timestamps = tobii_timestamps - start_time_tobii

        #多传感器时间轴人工补偿 (Time Offset)
        TIME_OFFSET = 0.40
        tobii_timestamps = tobii_timestamps + TIME_OFFSET
        
        # 3. 按照物理时间（而不是盲目的索引）进行精确对齐切片
        self.samples = self._create_samples_by_time(self.events, tobii_timestamps, self.raw_labels)

    def _create_samples_by_time(self, events, tobii_times, labels, window_duration=0.1):
        """
        按照物理时间进行滑窗对齐（默认窗口大小为 0.1 秒 = 100 毫秒）
        """
        samples = []
        max_time = events[-1, 2]
        num_windows = int(max_time / window_duration)
        
        counts = {0: 0, 1: 0} # 统计真实的类别数量
        
        print(f"正在按 {window_duration} 秒的时间窗口进行精确对齐，请稍候...")
        for i in range(num_windows):
            t_start = i * window_duration
            t_end = (i + 1) * window_duration
            
            # 截取这 0.1 秒内的所有事件
            mask = (events[:, 2] >= t_start) & (events[:, 2] < t_end)
            window_events = events[mask]
            
            # 如果这 0.1 秒内事件太少（眼球可能没动），跳过它以减少噪声
            if len(window_events) < 50:
                continue
                
            # 找到这 0.05 秒的中心时间点
            t_center = (t_start + t_end) / 2.0
            
            # 在 Tobii 的时间轴上，找到离这个中心时间最近的那个真实标签
            idx = (np.abs(tobii_times - t_center)).argmin()
            label = labels[idx]
            
            # 👑 【新增绝对核心】：时间特征归一化
            # 为了不污染原始 events 数组，我们 copy 一份当前窗口的数据
            window_events_norm = window_events.copy()
            # 1. 减去当前窗口的第一个时间戳，让时间从 0 开始起步 (比如 0.000 ~ 0.049)
            window_events_norm[:, 2] = window_events_norm[:, 2] - window_events_norm[0, 2]
            # 2. 除以窗口总长度，强制把时间特征压缩到 0 ~ 1 之间！
            window_events_norm[:, 2] = window_events_norm[:, 2] / window_duration
            
            # 把归一化后的干净数据放进样本库
            samples.append((window_events_norm, label))
            counts[label] += 1
            
        print(f"✅ 精确对齐完成！最终可用样本数: {len(samples)}")
        print(f"📊 真实数据分布 -> 注视(0): {counts[0]} 个, 扫视(1): {counts[1]} 个")
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_slice, label = self.samples[idx]
        
        # 绕过黑盒，使用均匀采样
        if len(event_slice) > self.seq_len:
            indices = np.linspace(0, len(event_slice) - 1, self.seq_len).astype(int)
            final_events = event_slice[indices].copy() # ⚠️必须加 .copy() 防止污染原始数据！
        elif len(event_slice) < self.seq_len:
            pad_len = self.seq_len - len(event_slice)
            padding = np.zeros((pad_len, 4))
            final_events = np.vstack((event_slice, padding))
        else:
            final_events = event_slice.copy()

        # 把当前这 50 毫秒窗口的第一个点的 X 和 Y 记为起点
        start_x = final_events[0, 0]
        start_y = final_events[0, 1]
        
        # 让后续所有的点都减去这个起点，强行把所有眼动轨迹拉回原点 (0,0)！
        final_events[:, 0] = final_events[:, 0] - start_x
        final_events[:, 1] = final_events[:, 1] - start_y
        # ==========================================

        # 计算相邻事件点之间的欧几里得距离（在去中心化后）
        x = final_events[:, 0]
        y = final_events[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.zeros(self.seq_len)
        distances[1:] = np.sqrt(dx*dx + dy*dy)
        
        # 添加第5个通道
        final_events = np.column_stack((final_events, distances))
        
        # 转置矩阵以适应 1D CNN: (通道数 5, 序列长度 seq_len)
        feature_matrix = final_events.T 
        
        feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return feature_tensor, label_tensor

def get_real_dataloaders(events_path, labels_path, batch_size=16, seq_len=100):
    """
    这一个大文件 80% 作为训练，20% 作为验证
    """
    dataset = EVEyeDataset(events_path, labels_path, seq_len=seq_len)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader