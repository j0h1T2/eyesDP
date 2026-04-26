import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import f1_score, precision_score, recall_score

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    模型训练与评估的主循环
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")
    model.to(device)

    print("在扫描训练集以计算类别动态权重...")
    all_train_labels = []
    # 这里只扫标签，不把数据放进 GPU，速度很快
    for _, batch_labels in train_loader:
        all_train_labels.extend(batch_labels.numpy())
        
    class_counts = np.bincount(all_train_labels)
    total_samples = len(all_train_labels)
    num_classes = len(class_counts)
    
    # 核心公式：总样本数 / (类别数 * 该类样本数)
    # 样本越少的类别，计算出的权重越大
    weights = total_samples / (num_classes * class_counts.astype(np.float32))
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"✅ 权重分配完毕 -> 注视(0): {weights[0]:.4f} | 扫视(1): {weights[1]:.4f}")
    # ==========================================
    
    # 将计算好的权重传入交叉熵损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 记录最好的 F1 分数，用于保存模型
    best_f1 = 0.0
    
    for epoch in range(epochs):
        # -----------------------------
        # 1. 训练阶段 (Training)
        # -----------------------------
        model.train()
        total_loss = 0.0
        
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 每隔几个 batch 打印一次训练进度，防止控制台看起来像死机了
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # -----------------------------
        # 2. 验证阶段 (Validation)
        # -----------------------------
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                
                outputs = model(val_data)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
        
        # -----------------------------
        # 3. 计算评估指标 (Metrics)
        # -----------------------------
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print("-" * 50)
        print(f"Epoch [{epoch+1}/{epochs}] 总结:")
        print(f"训练集平均 Loss: {avg_train_loss:.4f}")
        print(f"验证集评估 -> F1-Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        # -----------------------------
        # 4. 保存表现最好的模型
        # -----------------------------
        if f1 > best_f1:
            best_f1 = f1
            # 将模型权重保存到项目根目录下
            torch.save(model.state_dict(), 'best_event_eye_tracker.pth')
            print(f"🌟 发现更好的模型！验证集 F1: {best_f1:.4f}，已保存至 best_event_eye_tracker.pth")
        print("-" * 50)
        
    print("模型训练全部完成！\n best_f1=", best_f1)