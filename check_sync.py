import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# 配置中文字体，防止图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("正在加载数据，请稍候...")
    events_file = os.path.join('data', 'train', 'user1_s1_left_events.txt')
    labels_file = os.path.join('data', 'train', 'user1_s1_labels.txt')
    
    # 1. 读取事件相机数据
    df_events = pd.read_csv(events_file, sep='\s+', header=None)
    event_times = df_events[0].values
    event_times = (event_times - event_times[0]) / 1000000.0 # 转为秒
    
    # 计算每 0.1 秒的事件数量
    bins = np.arange(0, event_times[-1], 0.1)
    event_counts, _ = np.histogram(event_times, bins=bins)
    
    # 2. 读取 Tobii 数据并计算速度
    t_times, t_coords = [], []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                if item.get('type') == 'gaze' and item.get('data', {}).get('gaze2d'):
                    t_times.append(item['timestamp'])
                    t_coords.append(item['data']['gaze2d'])
            except: continue
            
    t_times = np.array(t_times)
    t_times = t_times - t_times[0] # 转为秒
    t_coords = np.array(t_coords)
    
    velocities = [0.0]
    for i in range(1, len(t_coords)):
        dt = t_times[i] - t_times[i-1]
        dist = np.linalg.norm(t_coords[i] - t_coords[i-1])
        velocities.append(dist / dt if dt > 1e-6 else 0.0)
        
    # 3. 画图对比（只画前 20 秒，看得很清楚）
    plt.figure(figsize=(15, 6))
    
    # 画事件数量波峰 (蓝色)
    plt.plot(bins[:-1], event_counts / np.max(event_counts), label='事件相机 (活跃度)', color='blue', alpha=0.7)
    
    # 画 Tobii 速度波峰 (红色)
    plt.plot(t_times, np.array(velocities) / np.max(velocities), label='Tobii 眼动仪 (移动速度)', color='red', alpha=0.7)
    
    plt.xlim(0, 20) # 只看前 20 秒
    plt.title("硬件时间同步检查 (如果波峰没有重合，说明时间轴错位！)")
    plt.xlabel("时间 (秒)")
    plt.ylabel("归一化强度")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()