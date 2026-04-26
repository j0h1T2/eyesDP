import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
from scipy.signal import medfilt
from .preprocess import EventDataPreprocessor

def parse_tobii_gaze(gaze_file_path):
    timestamps = []
    gaze_coords = []
    with open(gaze_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if item.get('type') == 'gaze':
                    data_dict = item.get('data', {})
                    if 'gaze2d' in data_dict and data_dict['gaze2d'] is not None:
                        timestamps.append(item['timestamp'])
                        gaze_coords.append(data_dict['gaze2d'])
            except json.JSONDecodeError:
                continue
    return np.array(timestamps), np.array(gaze_coords)

def calculate_ivt_labels(timestamps, coords, velocity_threshold=0.20):
    # ... (保持你原来的代码不变) ...
    if len(coords) < 2:
        return np.zeros(len(coords), dtype=int)
    velocities = [0.0]
    for i in range(1, len(coords)):
        dt = timestamps[i] - timestamps[i-1]
        dist = np.linalg.norm(coords[i] - coords[i-1])
        velocities.append(dist / dt if dt > 1e-6 else 0.0)
    labels = (np.array(velocities) > velocity_threshold).astype(int)
    if len(labels) >= 7:
        labels = medfilt(labels, kernel_size=7).astype(int)
    return labels

class EVEyeDataset(Dataset):
    # 🌟 新增了用于消融实验和数据扩增的开关参数
    def __init__(self, events_file, labels_file, seq_len=100, 
                 is_train=False, use_zero_center=True, use_v_channel=True):
        super(EVEyeDataset, self).__init__()
        self.seq_len = seq_len
        self.is_train = is_train
        self.use_zero_center = use_zero_center
        self.use_v_channel = use_v_channel

        # 初始化小波预处理器
        self.preprocessor = EventDataPreprocessor(sensor_size=(260, 346))
        
        print(f"正在读取事件文件: {events_file} ...")
        df_events = pd.read_csv(events_file, sep='\s+', header=None)
        raw_events = df_events.values
        
        self.events = np.zeros((len(raw_events), 4))
        self.events[:, 0] = raw_events[:, 1] / 346.0  # X
        self.events[:, 1] = raw_events[:, 2] / 260.0  # Y
        self.events[:, 2] = raw_events[:, 0]          # Timestamp
        self.events[:, 3] = (raw_events[:, 3] * 2) - 1.0 # Polarity (-1, 1)

        print(f"正在读取标签文件: {labels_file} ...")
        tobii_timestamps, tobii_coords = parse_tobii_gaze(labels_file)
        self.raw_labels = calculate_ivt_labels(tobii_timestamps, tobii_coords)
        
        # 统一时间轴
        start_time_micro = self.events[0, 2]
        self.events[:, 2] = (self.events[:, 2] - start_time_micro) / 1000000.0
        
        start_time_tobii = tobii_timestamps[0]
        tobii_timestamps = tobii_timestamps - start_time_tobii

        # 时间轴物理补偿
        TIME_OFFSET = 0.40
        tobii_timestamps = tobii_timestamps + TIME_OFFSET
        
        self.samples = self._create_samples_adaptive(self.events, tobii_timestamps, self.raw_labels, target_event_count=2000)

    def _create_samples_adaptive(self, events, tobii_times, labels, target_event_count=2000):
        """
        🌟 创新点：基于眼球运动速率的自适应聚合机制 (Constant Event Count)
        不再使用死板的 50ms，而是根据事件堆积的密度动态伸缩时间窗口
        """
        from tqdm import tqdm
        import numpy as np
        
        samples = []
        counts = {0: 0, 1: 0}
        
        BLINK_DENSITY_THRESHOLD = 126671 
        blink_drop_count = 0
        noise_drop_count = 0
        
        print(f"🚀 启动自适应动态聚合机制，目标密度: {target_event_count} 事件/窗口...")
        
        num_events = len(events)
        i = 0
        
        # 使用 tqdm 追踪事件处理进度
        with tqdm(total=num_events, desc="动态聚合进度", unit="event") as pbar:
            while i < num_events:
                start_idx = i
                # 动态划分：往后数 target_event_count 个事件作为一个窗口
                end_idx = min(i + target_event_count, num_events)
                
                window_events = events[start_idx:end_idx]
                
                if len(window_events) < 10:
                    break # 剩余事件不足，退出
                
                # 动态计算当前窗口的真实时间跨度 (可能只有 2ms，也可能有 200ms)
                t_start = window_events[0, 2]
                t_end = window_events[-1, 2]
                current_window_duration = t_end - t_start
                
                # ==========================================
                # 🛡️ 智能拦截与异常过滤
                # ==========================================
                # 1. 硬件风暴拦截：极短时间内涌入几十万事件（极速眨眼或硬件毛刺）
                if current_window_duration < 0.001 and len(window_events) > BLINK_DENSITY_THRESHOLD:
                    blink_drop_count += 1
                    i = end_idx
                    pbar.update(len(window_events))
                    continue
                    
                # 2. 静默底噪拦截：花了大半秒甚至几秒才凑齐这些事件，说明眼球根本没动，全是底噪
                if current_window_duration > 0.5:
                    noise_drop_count += 1
                    i = end_idx
                    pbar.update(len(window_events))
                    continue
                # ==========================================
                
                # 对齐物理标签 (找当前动态窗口的中心时间点)
                t_center = (t_start + t_end) / 2.0
                idx = (np.abs(tobii_times - t_center)).argmin()
                label = labels[idx]
                
                # 🌟 核心对齐：时间特征归一化
                # 无论这个窗口实际跨越了 5ms 还是 50ms，都将其映射到 [0, 1] 区间
                # 这样 LSTM 接收到的始终是标准的动态演化过程
                window_events_norm = window_events.copy()
                window_events_norm[:, 2] = window_events_norm[:, 2] - t_start
                if current_window_duration > 1e-6:
                    window_events_norm[:, 2] = window_events_norm[:, 2] / current_window_duration
                
                samples.append((window_events_norm, label, current_window_duration))
                counts[label] += 1
                
                # 步进到下一组事件
                i = end_idx
                pbar.update(len(window_events))

        print(f"\n🛡️ 拦截器报告：清除 {blink_drop_count} 个风暴窗口，过滤 {noise_drop_count} 个长时底噪窗口。")
        print(f"✅ 动态聚合完成！最终可用样本数: {len(samples)}")
        print(f"📊 真实数据分布 -> 注视(0): {counts[0]} 个, 扫视(1): {counts[1]} 个")
        
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_slice, label, _ = self.samples[idx]
        
        if len(event_slice) > self.seq_len:
            indices = np.linspace(0, len(event_slice) - 1, self.seq_len).astype(int)
            final_events = event_slice[indices].copy()
        elif len(event_slice) < self.seq_len:
            pad_len = self.seq_len - len(event_slice)
            padding = np.zeros((pad_len, 4))
            final_events = np.vstack((event_slice, padding))
        else:
            final_events = event_slice.copy()

        # ==========================================
        # 🔬 消融实验与扩增开关
        # ==========================================
        if self.use_zero_center:
            start_x = final_events[0, 0]
            start_y = final_events[0, 1]
            final_events[:, 0] = final_events[:, 0] - start_x
            final_events[:, 1] = final_events[:, 1] - start_y
            
        if self.is_train and self.use_zero_center:
            # 仅在去中心化的情况下进行物理翻转和抖动扩增
            if np.random.rand() < 0.5:
                final_events[:, 0] = -final_events[:, 0] 
            if np.random.rand() < 0.5:
                final_events[:, 1] = -final_events[:, 1] 
            if np.random.rand() < 0.5:
                final_events[:, 0] += np.random.normal(0, 0.015, size=self.seq_len)
                final_events[:, 1] += np.random.normal(0, 0.015, size=self.seq_len)

        if self.use_v_channel:
            x = final_events[:, 0]
            y = final_events[:, 1]
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.zeros(self.seq_len)
            distances[1:] = np.sqrt(dx*dx + dy*dy)
        else:
            # 如果关闭 V 通道，填入 0，以保持模型 5 通道输入结构不崩溃
            distances = np.zeros(self.seq_len)

        #新增：对 X 和 Y 坐标轨迹也进行小波平滑 (大幅抑制低信噪比带来的干扰)
        final_events[:, 0] = self.preprocessor.wavelet_denoise(final_events[:, 0])
        final_events[:, 1] = self.preprocessor.wavelet_denoise(final_events[:, 1])

        final_events = np.column_stack((final_events, distances))
        feature_matrix = final_events.T

        return torch.tensor(feature_matrix, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_real_dataloaders(events_path, labels_path, batch_size=16, seq_len=100):
    # 分别实例化，确保验证集绝对不开启数据扩增
    full_train_dataset = EVEyeDataset(events_path, labels_path, seq_len=seq_len, is_train=True)
    full_val_dataset = EVEyeDataset(events_path, labels_path, seq_len=seq_len, is_train=False)
    
    dataset_length = len(full_train_dataset)
    train_size = int(0.8 * dataset_length)
    val_size = dataset_length - train_size
    
    torch.manual_seed(42) # 固定随机种子，防止评估分数波动
    indices = torch.randperm(dataset_length).tolist()
    
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(full_val_dataset, indices[train_size:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader