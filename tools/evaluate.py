import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report

# 👑 关键步骤：将项目根目录添加到系统路径，以便在 tools 文件夹内也能找到 core 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import EventEyeTrackerModel
from core.dataset import get_real_dataloaders

def run_evaluation():
    # 1. 路径配置
    # 注意：因为脚本在 tools 文件夹运行，.. 代表返回上一级根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "best_event_eye_tracker.pth") 
    events_path = os.path.join(base_dir, "data", "train", "user1_s1_left_events.txt")
    labels_path = os.path.join(base_dir, "data", "train", "user1_s1_labels.txt")

    # 2. 设备与数据准备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")
    
    # 获取数据加载器
    _, val_loader = get_real_dataloaders(events_path, labels_path, batch_size=32)
    
    # 3. 加载模型
    model = EventEyeTrackerModel(input_channels=5, cnn_out_channels=16, lstm_hidden=32, num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功从根目录加载权重: {model_path}")
    else:
        print(f"❌ 错误：在 {model_path} 找不到模型文件，请确认 .pth 文件在项目根目录下")
        return

    model.eval()
    all_preds = []
    all_labels = []

    # 4. 执行推理
    print("正在运行验证集推理...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- 可视化 1：混淆矩阵 ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fixation (0)', 'Saccade (1)'], 
                yticklabels=['Fixation (0)', 'Saccade (1)'])
    plt.title('Confusion Matrix: Eye Event Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # 保存到 tools/ 文件夹下
    plt.savefig(os.path.join(os.path.dirname(__file__), 'report_confusion_matrix.png'), dpi=300)
    plt.show()

    # --- 可视化 2：时间轴同步对比图 ---
    plt.figure(figsize=(16, 6))
    plot_range = 300 
    plt.step(range(plot_range), all_labels[:plot_range], label='Ground Truth (Tobii)', color='#1f77b4', where='post', lw=2)
    plt.step(range(plot_range), all_preds[:plot_range], label='Model Prediction', color='#ff7f0e', linestyle='--', where='post', lw=2)
    plt.fill_between(range(plot_range), 0, all_labels[:plot_range], color='#1f77b4', alpha=0.1)
    plt.title(f'Temporal Alignment Analysis (Top {plot_range} samples)')
    plt.xlabel('Time Window Index')
    plt.ylabel('Event Type')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'report_alignment_comparison.png'), dpi=300)
    plt.show()

    # 5. 输出指标
    print("\n" + "="*60)
    print(classification_report(all_labels, all_preds, target_names=['Fixation', 'Saccade']))
    print("="*60)
    print(f"📈 图片已保存至 tools 文件夹下。")

if __name__ == "__main__":
    run_evaluation()