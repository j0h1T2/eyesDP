import numpy as np
import pywt
import torch

class EventDataPreprocessor:
    def __init__(self, sensor_size=(260, 346), noise_threshold=0.5):
        """
        sensor_size: EV-Eye 数据集使用的 DAVIS346 分辨率为 346x260 (W x H)
        """
        self.height = sensor_size[0]
        self.width = sensor_size[1]
        self.noise_threshold = noise_threshold

    def wavelet_denoise(self, signal, wavelet='db4', level=2):
        """利用小波变换进行去噪处理"""
        # 【新增防爆代码】如果这段时间没有任何事件发生（全0），直接跳过去噪
        if np.all(signal == 0):
            return signal
            
        coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # 【新增防爆代码】防止 sigma 为 0 导致后续阈值计算出错
        if sigma == 0:
            return signal
            
        uthresh = sigma * np.sqrt(2 * np.log(len(signal) + 1e-5))
        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
        denoised_signal = pywt.waverec(coeffs, wavelet, mode='per')
        return denoised_signal[:len(signal)]

    def events_to_time_surfaces(self, events, num_bins=50):
        """
        核心难点：将异步事件流 (N, 4) 聚合成等长的时间序列特征 (Channels, Seq_Len)
        events: numpy array, shape (N, 4) -> (x, y, timestamp, polarity)
        """
        if len(events) == 0:
            # 考虑模型输入为 (in_channels=4, seq_len=num_bins)
            return np.zeros((4, num_bins), dtype=np.float32)

        t_start = events[0, 2]
        t_end = events[-1, 2]
        time_window = (t_end - t_start) / num_bins
        
        # 提取 4 个维度的全局特征（举例：正极性事件数、负极性事件数、X轴重心、Y轴重心）
        # 这样可以将二维的图像流降维成一维时序信号，喂给 1D-CNN
        features = np.zeros((4, num_bins), dtype=np.float32)

        for i in range(num_bins):
            # 找到当前时间窗口内的事件
            t_curr_start = t_start + i * time_window
            t_curr_end = t_start + (i + 1) * time_window
            
            mask = (events[:, 2] >= t_curr_start) & (events[:, 2] < t_curr_end)
            window_events = events[mask]
            
            if len(window_events) > 0:
                pos_events = window_events[window_events[:, 3] == 1]
                neg_events = window_events[window_events[:, 3] == 0]
                
                # 特征 1: 正极性事件数量 (并进行简单的对数压缩防止过大)
                features[0, i] = np.log(len(pos_events) + 1)
                # 特征 2: 负极性事件数量
                features[1, i] = np.log(len(neg_events) + 1)
                # 特征 3: 活跃事件的平均 X 坐标归一化
                features[2, i] = np.mean(window_events[:, 0]) / self.width
                # 特征 4: 活跃事件的平均 Y 坐标归一化
                features[3, i] = np.mean(window_events[:, 1]) / self.height

        # 对提取的时序特征应用小波去噪 (可选)
        for c in range(4):
            features[c, :] = self.wavelet_denoise(features[c, :])

        return features