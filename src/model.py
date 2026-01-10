"""
CNN 卷积神经网络模型定义
用于 MNIST 手写数字识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    卷积神经网络模型
    
    网络结构:
    - 输入: 28x28 灰度图像
    - Conv1: 32个 3x3 卷积核
    - Conv2: 64个 3x3 卷积核
    - FC1: 128 个神经元
    - FC2: 10 个输出（0-9）
    """
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # 第一层卷积: 1通道 -> 32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二层卷积: 32通道 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 28/2/2 = 7, 所以 FC 输入是 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """前向传播"""
        # 第一层卷积 + BN + ReLU + 池化
        # 输入: (batch, 1, 28, 28) -> 输出: (batch, 32, 14, 14)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二层卷积 + BN + ReLU + 池化
        # 输入: (batch, 32, 14, 14) -> 输出: (batch, 64, 7, 7)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """预测函数，返回类别和概率"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            prob = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
        return pred, prob


def get_model():
    """获取模型实例"""
    return CNN()


if __name__ == "__main__":
    # 测试模型
    model = CNN()
    print(model)
    
    # 测试输入
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
