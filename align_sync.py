#!/usr/bin/env python3
"""
align_sync.py - 自动计算事件相机和 Tobii 眼动仪之间的毫秒级精确时间偏移量

步骤：
1. 读取 data/train/user1_s1_left_events.txt 和 data/train/user1_s1_labels.txt
2. 将事件相机的活跃度（Event Count）和 Tobii 眼动仪的速度（Velocity）重采样到同一个固定的时间网格上（5ms bins）
3. 使用 scipy.signal.correlate (互相关算法) 计算这两个时间序列的互相关性
4. 找到互相关峰值（argmax）对应的延迟时间（Lag）
5. 打印出最精确的 TIME_OFFSET 值
6. 使用 matplotlib 画出偏移补偿前和补偿后的对齐对比图
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy import signal

# 配置中文字体，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_event_camera_data(filepath):
    """
    加载事件相机数据，计算活跃度（每5ms的事件计数）
    
    参数:
        filepath: 事件相机数据文件路径
        
    返回:
        time_grid: 时间网格（秒）
        event_counts: 对应时间网格的事件计数
    """
    print(f"正在加载事件相机数据: {filepath}")
    df_events = pd.read_csv(filepath, sep='\s+', header=None)
    
    # 第一列是时间戳（微秒）
    event_times_microseconds = df_events[0].values
    
    # 转换为秒，并从第一个事件开始计时
    event_times_seconds = (event_times_microseconds - event_times_microseconds[0]) / 1_000_000.0
    
    # 总持续时间
    total_duration = event_times_seconds[-1] - event_times_seconds[0]
    
    # 创建5ms (0.005秒) 的时间网格
    bin_size = 0.005  # 5ms
    num_bins = int(total_duration / bin_size) + 1
    time_grid = np.arange(0, total_duration, bin_size)
    
    # 计算每个bin中的事件数量
    event_counts, _ = np.histogram(event_times_seconds, bins=time_grid)
    
    # 确保长度一致
    if len(event_counts) > len(time_grid):
        event_counts = event_counts[:-1]
    
    print(f"  事件数量: {len(event_times_seconds)}")
    print(f"  持续时间: {total_duration:.2f} 秒")
    print(f"  时间网格大小: {len(time_grid)} (每 {bin_size*1000}ms)")
    
    return time_grid[:len(event_counts)], event_counts

def load_tobii_data(filepath):
    """
    加载Tobii眼动仪数据，计算速度（每5ms的速度）
    
    参数:
        filepath: Tobii数据文件路径
        
    返回:
        time_grid: 时间网格（秒）
        velocities: 对应时间网格的速度
    """
    print(f"正在加载Tobii眼动仪数据: {filepath}")
    
    t_times, t_coords = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if item.get('type') == 'gaze' and item.get('data', {}).get('gaze2d'):
                    t_times.append(item['timestamp'])
                    t_coords.append(item['data']['gaze2d'])
            except:
                continue
    
    if not t_times:
        raise ValueError("未找到有效的Tobii眼动数据")
    
    t_times = np.array(t_times)
    t_coords = np.array(t_coords)
    
    # 时间戳已经是秒，从第一个时间戳开始计时
    t_times = t_times - t_times[0]
    
    # 计算速度
    velocities = np.zeros(len(t_times))
    for i in range(1, len(t_times)):
        dt = t_times[i] - t_times[i-1]
        if dt > 1e-6:
            dist = np.linalg.norm(t_coords[i] - t_coords[i-1])
            velocities[i] = dist / dt
        else:
            velocities[i] = 0.0
    
    # 总持续时间
    total_duration = t_times[-1] - t_times[0]
    
    # 创建5ms (0.005秒) 的时间网格
    bin_size = 0.005  # 5ms
    time_grid = np.arange(0, total_duration, bin_size)
    
    # 将速度重采样到时间网格上（使用线性插值）
    from scipy.interpolate import interp1d
    
    # 创建插值函数
    interp_func = interp1d(t_times, velocities, kind='linear', 
                           bounds_error=False, fill_value=0.0)
    
    # 在时间网格上插值
    resampled_velocities = interp_func(time_grid)
    
    # 用相邻值填充NaN
    resampled_velocities = np.nan_to_num(resampled_velocities, nan=0.0)
    
    print(f"  Tobii数据点数量: {len(t_times)}")
    print(f"  持续时间: {total_duration:.2f} 秒")
    print(f"  重采样后速度大小: {len(resampled_velocities)}")
    
    return time_grid[:len(resampled_velocities)], resampled_velocities

def compute_cross_correlation(signal1, signal2, time_grid, max_lag_seconds=2.0):
    """
    计算两个信号的互相关，找出最佳时间偏移
    
    参数:
        signal1: 第一个信号（事件相机活跃度）
        signal2: 第二个信号（Tobii速度）
        time_grid: 时间网格
        max_lag_seconds: 最大延迟搜索范围（秒）
        
    返回:
        lag_times: 延迟时间数组（秒）
        correlation: 互相关值数组
        best_lag: 最佳延迟时间（秒）
        best_lag_ms: 最佳延迟时间（毫秒）
    """
    print("正在计算互相关...")
    
    # 确保信号长度相同
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    
    # 归一化信号（减去均值，除以标准差）
    signal1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    signal2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
    
    # 计算互相关
    correlation = signal.correlate(signal1_norm, signal2_norm, mode='full')
    
    # 创建延迟时间轴
    dt = time_grid[1] - time_grid[0]  # 时间步长（0.005秒）
    lag_samples = np.arange(-len(signal2_norm) + 1, len(signal1_norm))
    lag_times = lag_samples * dt
    
    # 限制在最大延迟范围内
    max_lag_samples = int(max_lag_seconds / dt)
    center_idx = len(correlation) // 2
    start_idx = max(0, center_idx - max_lag_samples)
    end_idx = min(len(correlation), center_idx + max_lag_samples + 1)
    
    lag_times_limited = lag_times[start_idx:end_idx]
    correlation_limited = correlation[start_idx:end_idx]
    
    # 找到最佳延迟
    best_idx = np.argmax(correlation_limited)
    best_lag = lag_times_limited[best_idx]
    best_lag_ms = best_lag * 1000  # 转换为毫秒
    
    print(f"  互相关计算完成，搜索范围: ±{max_lag_seconds}秒")
    print(f"  最佳延迟: {best_lag:.6f} 秒 ({best_lag_ms:.2f} 毫秒)")
    print(f"  最大互相关值: {correlation_limited[best_idx]:.6f}")
    
    return lag_times_limited, correlation_limited, best_lag, best_lag_ms

def plot_alignment_comparison(event_time, event_signal, tobii_time, tobii_signal, 
                             time_offset_ms, time_offset_seconds):
    """
    绘制偏移补偿前和补偿后的对齐对比图
    
    参数:
        event_time: 事件相机时间网格
        event_signal: 事件相机信号
        tobii_time: Tobii时间网格
        tobii_signal: Tobii信号
        time_offset_ms: 时间偏移（毫秒）
        time_offset_seconds: 时间偏移（秒）
    """
    print("正在生成对齐对比图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. 偏移补偿前的对比
    ax1 = axes[0]
    
    # 归一化信号以便更好地比较
    event_norm = (event_signal - np.mean(event_signal)) / (np.std(event_signal) + 1e-10)
    tobii_norm = (tobii_signal - np.mean(tobii_signal)) / (np.std(tobii_signal) + 1e-10)
    
    # 只显示前30秒以便清晰查看
    max_time = min(30, max(event_time[-1], tobii_time[-1]))
    mask1 = event_time <= max_time
    mask2 = tobii_time <= max_time
    
    ax1.plot(event_time[mask1], event_norm[mask1], 
             label='事件相机 (活跃度)', color='blue', alpha=0.7, linewidth=1.5)
    ax1.plot(tobii_time[mask2], tobii_norm[mask2], 
             label='Tobii 眼动仪 (速度)', color='red', alpha=0.7, linewidth=1.5)
    
    ax1.set_xlim(0, max_time)
    ax1.set_title(f'偏移补偿前 | 时间偏移: {time_offset_ms:.2f} ms ({time_offset_seconds:.4f} s)')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('归一化强度')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. 偏移补偿后的对比
    ax2 = axes[1]
    
    # 应用时间偏移补偿
    if time_offset_seconds > 0:
        # 事件相机延迟（需要将事件相机数据向后移动）
        shift_samples = int(time_offset_seconds / (event_time[1] - event_time[0]))
        if shift_samples < len(event_norm):
            event_shifted = np.roll(event_norm, shift_samples)
            event_shifted[:shift_samples] = 0
        else:
            event_shifted = event_norm
    else:
        # Tobii延迟（需要将Tobii数据向前移动）
        shift_samples = int(abs(time_offset_seconds) / (tobii_time[1] - tobii_time[0]))
        if shift_samples < len(tobii_norm):
            tobii_shifted = np.roll(tobii_norm, -shift_samples)
            tobii_shifted[-shift_samples:] = 0
        else:
            tobii_shifted = tobii_norm
    
    ax2.plot(event_time[mask1], event_norm[mask1], 
             label='事件相机 (活跃度)', color='blue', alpha=0.7, linewidth=1.5)
    
    if time_offset_seconds > 0:
        ax2.plot(event_time[mask1], event_shifted[mask1], 
                 label=f'事件相机 (向后移动 {time_offset_ms:.2f} ms)', 
                 color='cyan', alpha=0.9, linewidth=2, linestyle='--')
    else:
        ax2.plot(tobii_time[mask2], tobii_shifted[mask2], 
                 label=f'Tobii (向前移动 {abs(time_offset_ms):.2f} ms)', 
                 color='orange', alpha=0.9, linewidth=2, linestyle='--')
    
    ax2.set_xlim(0, max_time)
    ax2.set_title('偏移补偿后')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('归一化强度')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'time_alignment_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  对比图已保存至: {output_path}")
    
    plt.show()

def main():
    print("=" * 70)
    print("事件相机与Tobii眼动仪时间同步校准工具")
    print("=" * 70)
    
    try:
        # 1. 加载数据
        events_file = os.path.join('data', 'train', 'user1_s1_left_events.txt')
        labels_file = os.path.join('data', 'train', 'user1_s1_labels.txt')
        
        # 检查文件是否存在
        if not os.path.exists(events_file):
            raise FileNotFoundError(f"事件相机数据文件不存在: {events_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Tobii数据文件不存在: {labels_file}")
        
        # 加载事件相机数据
        event_time, event_counts = load_event_camera_data(events_file)
        
        # 加载Tobii数据
        tobii_time, tobii_velocities = load_tobii_data(labels_file)
        
        # 2. 确保时间网格长度一致
        min_len = min(len(event_time), len(tobii_time))
        event_time = event_time[:min_len]
        event_counts = event_counts[:min_len]
        tobii_time = tobii_time[:min_len]
        tobii_velocities = tobii_velocities[:min_len]
        
        print(f"\n对齐后的数据长度: {min_len}")
        print(f"时间范围: 0 到 {event_time[-1]:.2f} 秒")
        
        # 3. 计算互相关和时间偏移
        max_lag = 2.0  # 最大搜索延迟2秒
        lag_times, correlation, best_lag, best_lag_ms = compute_cross_correlation(
            event_counts, tobii_velocities, event_time, max_lag_seconds=max_lag
        )
        
        # 4. 输出结果
        print("\n" + "=" * 70)
        print("时间同步结果:")
        print("=" * 70)
        
        # 解释正负延迟的含义
        if best_lag > 0:
            print(f"事件相机相对于Tobii延迟了 {abs(best_lag_ms):.2f} ms")
            print(f"  含义: 事件相机的时间戳比Tobii晚 {abs(best_lag_ms):.2f} ms")
            print(f"  建议: 将事件相机数据向后移动 {abs(best_lag_ms):.2f} ms")
        elif best_lag < 0:
            print(f"Tobii相对于事件相机延迟了 {abs(best_lag_ms):.2f} ms")
            print(f"  含义: Tobii的时间戳比事件相机晚 {abs(best_lag_ms):.2f} ms")
            print(f"  建议: 将Tobii数据向后移动 {abs(best_lag_ms):.2f} ms")
        else:
            print("两个设备的时间已经完美同步!")
        
        print(f"\n最精确的 TIME_OFFSET 值:")
        print(f"  {best_lag_ms:.2f} ms  ({best_lag:.6f} seconds)")
        
        # 5. 绘制互相关函数
        fig_corr, ax_corr = plt.subplots(figsize=(12, 5))
        
        ax_corr.plot(lag_times * 1000, correlation, 'b-', linewidth=2, label='互相关函数')
        ax_corr.axvline(x=best_lag_ms, color='r', linestyle='--', 
                        linewidth=2, label=f'最佳延迟: {best_lag_ms:.2f} ms')
        
        ax_corr.set_xlim(-max_lag * 1000 / 2, max_lag * 1000 / 2)
        ax_corr.set_title('互相关函数 (峰值位置指示最佳时间偏移)')
        ax_corr.set_xlabel('延迟时间 (毫秒)')
        ax_corr.set_ylabel('互相关值')
        ax_corr.legend(loc='upper right')
        ax_corr.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_correlation.png', dpi=150, bbox_inches='tight')
        print("  互相关图已保存至: cross_correlation.png")
        plt.show()
        
        # 6. 绘制对齐对比图
        plot_alignment_comparison(
            event_time, event_counts, 
            tobii_time, tobii_velocities,
            best_lag_ms, best_lag
        )
        
        print("\n" + "=" * 70)
        print("时间同步校准完成!")
        print("=" * 70)
        
        # 返回偏移值供其他脚本使用
        return best_lag_ms, best_lag
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查:")
        print("  1. 数据文件路径是否正确")
        print("  2. 数据文件格式是否正确")
        print("  3. 是否安装了必要的Python包 (pandas, numpy, scipy, matplotlib)")
        return None, None

if __name__ == "__main__":
    main()