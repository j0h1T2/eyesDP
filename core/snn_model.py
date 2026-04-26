import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SpikingEyeTracker(nn.Module):
    def __init__(self, input_channels=5, hidden_channels=16, num_classes=2):
        super().__init__()
        # 代理梯度，用于跨越离散脉冲不可导的障碍
        spike_grad = surrogate.atan()
        # 膜电位衰减率 (Leak Rate)，充当短时记忆
        beta = 0.8
        
        # 空间特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels)
        )
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        
        # 全连接分类
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
        # 将输入重构为 (Time_Steps, Batch, Channels)
        x = x.permute(2, 0, 1) 
        time_steps = x.size(0)
        
        # 清空“漏水桶”的初始膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = [] # 记录输出脉冲
        
        # 沿着时间步推进 (事件驱动核心)
        for step in range(time_steps):
            current_input = x[step].unsqueeze(-1)*2.0
            cur_features = self.conv(current_input).squeeze(-1)
            
            # 神经元点火逻辑
            spk1, mem1 = self.lif1(cur_features, mem1)
            cur_fc = self.fc(spk1)
            spk2, mem2 = self.lif2(cur_fc, mem2)
            
            spk2_rec.append(spk2)
            
        # (Time_Steps, Batch, Num_Classes)
        spk2_rec = torch.stack(spk2_rec, dim=0)
        
        # 速率编码：累加200步内的脉冲总数，作为类似 Softmax 前的 Logits
        out_spikes = spk2_rec.sum(dim=0) 
        return out_spikes