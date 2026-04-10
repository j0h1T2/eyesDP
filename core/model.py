import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1)
        # Context vector shape: (batch_size, hidden_dim)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context_vector, attn_weights

class MultiScale1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScale1DCNN, self).__init__()
        # 多尺度卷积核设计，适应不同持续时间的眼动事件
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        
    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))
        # 拼接多尺度特征
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn(out)
        return out

class EventEyeTrackerModel(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, lstm_hidden, num_classes=3):
        super(EventEyeTrackerModel, self).__init__()
        # CNN 模块
        self.cnn = MultiScale1DCNN(input_channels, cnn_out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM 模块
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels * 3, 
            hidden_size=lstm_hidden, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = Attention(lstm_hidden * 2) # 双向 LSTM，维度乘 2
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        c_out = self.cnn(x)
        c_out = self.pool(c_out)
        
        # 转换维度以适应 LSTM: (batch_size, seq_len, channels)
        c_out = c_out.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(c_out)
        
        # 应用注意力机制
        attn_out, weights = self.attention(lstm_out)
        
        # 分类输出
        logits = self.classifier(attn_out)
        return logits