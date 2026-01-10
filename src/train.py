"""
MNIST 模型训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CNN


def get_data_loaders(data_dir, batch_size=64, augment=True):
    """
    获取训练和测试数据加载器
    
    Args:
        data_dir: 数据存放目录
        batch_size: 批次大小
        augment: 是否使用数据增强（提高对手写变化的鲁棒性）
    
    Returns:
        train_loader, test_loader
    """
    # 测试数据预处理（标准化）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if augment:
        # 训练数据增强 - 模拟真实手写的各种变化
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),  # 随机旋转 ±15°
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 随机平移 10%
                scale=(0.9, 1.1),      # 随机缩放 90%-110%
                shear=10               # 随机倾斜
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("✓ 已启用数据增强（旋转、平移、缩放、倾斜）")
    else:
        train_transform = test_transform
    
    # 下载并加载训练数据
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 下载并加载测试数据
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  批次 [{batch_idx + 1}/{len(train_loader)}] "
                  f"损失: {loss.item():.4f} "
                  f"准确率: {100. * correct / total:.2f}%")
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def train(epochs=10, batch_size=64, learning_rate=0.001):
    """
    训练模型
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置路径
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    model_dir = os.path.join(project_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 获取数据加载器
    print("正在加载数据集...")
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    
    # 创建模型
    model = CNN().to(device)
    print(f"\n模型结构:\n{model}\n")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 训练循环
    best_acc = 0.0
    print("开始训练...\n")
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        print(f"\n  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.2f}%\n")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(model_dir, "mnist_cnn_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, model_path)
            print(f"  ✓ 保存最佳模型 (准确率: {test_acc:.2f}%)\n")
    
    print("=" * 40)
    print(f"训练完成！最佳测试准确率: {best_acc:.2f}%")
    print(f"模型已保存至: {os.path.join(model_dir, 'mnist_cnn_best.pth')}")


if __name__ == "__main__":
    train(epochs=10, batch_size=64, learning_rate=0.001)
