"""
MNIST 预测/推理脚本
"""

import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CNN


def load_model(model_path=None):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径，默认使用最佳模型
    
    Returns:
        model: 加载好权重的模型
        device: 使用的设备
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "mnist_cnn_best.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 train.py 训练模型")
    
    model = CNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"已加载模型: {model_path}")
    print(f"模型准确率: {checkpoint['accuracy']:.2f}%")
    
    return model, device


def preprocess_image(image_path):
    """
    预处理图像
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        tensor: 预处理后的图像张量
    """
    # 数据预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载并处理图像
    image = Image.open(image_path)
    tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
    
    return tensor


def preprocess_canvas_image(image_data):
    """
    预处理来自画布的图像数据 - 增强版
    模拟 MNIST 的标准化处理：检测边界、居中、缩放
    
    Args:
        image_data: PIL Image 对象
    
    Returns:
        tensor: 预处理后的图像张量
    """
    import numpy as np
    from PIL import ImageOps
    
    # 转为灰度图
    img = image_data.convert('L')
    img_array = np.array(img)
    
    # 找到数字的边界框
    # 假设背景是白色(255)，数字是深色
    threshold = 200  # 阈值
    rows = np.any(img_array < threshold, axis=1)
    cols = np.any(img_array < threshold, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # 如果没有检测到内容，直接返回
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform(img).unsqueeze(0)
    
    # 获取边界
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # 裁剪数字区域
    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    # 保持长宽比，放入正方形
    width, height = cropped.size
    max_dim = max(width, height)
    
    # 创建正方形白色背景，留出边距
    padding = int(max_dim * 0.2)
    new_size = max_dim + 2 * padding
    square_img = Image.new('L', (new_size, new_size), 255)
    
    # 将裁剪的数字居中放置
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    square_img.paste(cropped, (x_offset, y_offset))
    
    # 调整到 28x28（MNIST 标准尺寸）
    resized = square_img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 反转颜色（MNIST 是黑底白字）
    inverted = ImageOps.invert(resized)
    
    # 转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(inverted).unsqueeze(0)
    return tensor


def predict(model, image_tensor, device):
    """
    预测图像中的数字
    
    Args:
        model: CNN 模型
        image_tensor: 图像张量
        device: 设备
    
    Returns:
        digit: 预测的数字
        confidence: 置信度
        probabilities: 各数字的概率
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        digit = predicted.item()
        confidence = confidence.item()
        probs = probabilities.squeeze().cpu().numpy()
    
    return digit, confidence, probs


def segment_digits(image):
    """
    分割图像中的多个数字
    
    使用垂直投影法来分离各个数字（更适合连续书写）
    
    Args:
        image: PIL Image 对象（白底黑字）
    
    Returns:
        digit_images: 按 X 坐标排序的数字图像列表
        bboxes: 各数字的边界框 [(x1, y1, x2, y2), ...]
    """
    import numpy as np
    from PIL import ImageOps
    
    # 转为灰度图
    img = image.convert('L')
    img_array = np.array(img)
    
    # 二值化（白底黑字 -> 0为背景，1为数字）
    threshold = 200
    binary = (img_array < threshold).astype(np.uint8)
    
    if not np.any(binary):
        return [], []
    
    height, width = binary.shape
    
    # 找到整体内容的边界
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [], []
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # 裁剪到内容区域
    content_binary = binary[y_min:y_max+1, x_min:x_max+1]
    content_height, content_width = content_binary.shape
    
    # 计算垂直投影（每列的像素数）
    vertical_projection = np.sum(content_binary, axis=0)
    
    # 找到分割点（投影值为0或很小的区域）
    # 使用动态阈值
    projection_threshold = max(1, np.max(vertical_projection) * 0.1)
    
    # 找到所有"空白"区域
    is_gap = vertical_projection < projection_threshold
    
    # 找到数字区间（非空白的连续区域）
    digit_ranges = []
    in_digit = False
    start = 0
    
    for i in range(content_width):
        if not is_gap[i] and not in_digit:
            # 开始新数字
            in_digit = True
            start = i
        elif is_gap[i] and in_digit:
            # 结束当前数字
            in_digit = False
            if i - start > 5:  # 过滤太窄的区域
                digit_ranges.append((start, i))
    
    # 处理最后一个数字
    if in_digit and content_width - start > 5:
        digit_ranges.append((start, content_width))
    
    # 如果只找到一个很宽的区域，尝试强制分割
    if len(digit_ranges) == 1:
        start, end = digit_ranges[0]
        region_width = end - start
        
        # 如果区域宽度超过高度的1.5倍，可能是多个数字
        if region_width > content_height * 1.5:
            # 估算数字个数
            estimated_digits = max(2, int(region_width / content_height))
            
            # 使用投影的局部最小值来分割
            segment_projection = vertical_projection[start:end]
            digit_ranges = find_valleys_for_split(segment_projection, start, estimated_digits)
    
    if not digit_ranges:
        # 如果分割失败，返回整个图像
        return [img], [(x_min, y_min, x_max, y_max)]
    
    # 裁剪每个数字区域
    digit_images = []
    bboxes = []
    
    for x_start, x_end in digit_ranges:
        # 转换回原始坐标
        abs_x1 = x_min + x_start
        abs_x2 = x_min + x_end
        
        # 添加边距
        pad = 3
        abs_x1 = max(0, abs_x1 - pad)
        abs_x2 = min(width, abs_x2 + pad)
        abs_y1 = max(0, y_min - pad)
        abs_y2 = min(height, y_max + pad)
        
        digit_img = img.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        digit_images.append(digit_img)
        bboxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
    
    return digit_images, bboxes


def find_valleys_for_split(projection, offset, num_splits):
    """
    在投影中找到谷值点用于分割
    
    Args:
        projection: 垂直投影数组
        offset: 偏移量
        num_splits: 期望的分割数
    
    Returns:
        ranges: 分割后的区间列表
    """
    import numpy as np
    
    length = len(projection)
    segment_width = length // num_splits
    
    ranges = []
    current_start = 0
    
    for i in range(num_splits - 1):
        # 在预期分割点附近找最小值
        search_start = max(0, (i + 1) * segment_width - segment_width // 3)
        search_end = min(length, (i + 1) * segment_width + segment_width // 3)
        
        if search_start < search_end:
            # 找到这个区域的最小值位置
            local_min_idx = search_start + np.argmin(projection[search_start:search_end])
            
            if local_min_idx > current_start + 5:
                ranges.append((offset + current_start, offset + local_min_idx))
                current_start = local_min_idx
    
    # 添加最后一个区间
    if length - current_start > 5:
        ranges.append((offset + current_start, offset + length))
    
    return ranges


def merge_close_bboxes(bboxes, threshold=15):
    """
    合并接近的边界框（处理同一数字的断开笔画）
    
    Args:
        bboxes: 边界框列表
        threshold: 合并阈值（像素）
    
    Returns:
        merged: 合并后的边界框列表
    """
    if not bboxes:
        return []
    
    # 按 X 坐标排序
    bboxes = sorted(bboxes, key=lambda b: b[0])
    
    merged = [list(bboxes[0])]
    
    for box in bboxes[1:]:
        last = merged[-1]
        # 检查是否应该合并（X 方向有重叠或距离很近）
        if box[0] <= last[2] + threshold:
            # 合并
            last[0] = min(last[0], box[0])
            last[1] = min(last[1], box[1])
            last[2] = max(last[2], box[2])
            last[3] = max(last[3], box[3])
        else:
            merged.append(list(box))
    
    return [tuple(b) for b in merged]


def preprocess_single_digit(digit_img):
    """
    预处理单个数字图像（用于多数字识别）
    
    Args:
        digit_img: 单个数字的 PIL Image（灰度，白底黑字）
    
    Returns:
        tensor: 预处理后的图像张量
    """
    import numpy as np
    from PIL import ImageOps
    
    # 保持长宽比，放入正方形
    width, height = digit_img.size
    max_dim = max(width, height)
    
    # 创建正方形白色背景，留出边距
    padding = int(max_dim * 0.2)
    new_size = max_dim + 2 * padding
    square_img = Image.new('L', (new_size, new_size), 255)
    
    # 居中放置
    x_offset = (new_size - width) // 2
    y_offset = (new_size - height) // 2
    square_img.paste(digit_img, (x_offset, y_offset))
    
    # 调整到 28x28
    resized = square_img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 反转颜色（MNIST 是黑底白字）
    inverted = ImageOps.invert(resized)
    
    # 转换为张量并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(inverted).unsqueeze(0)
    return tensor


def predict_multi_digits(model, image, device):
    """
    识别图像中的多个数字
    
    Args:
        model: CNN 模型
        image: PIL Image 对象（白底黑字的完整图像）
        device: 设备
    
    Returns:
        result: 识别结果字符串（如 "118"）
        details: 每个数字的详细信息列表
    """
    # 分割数字
    digit_images, bboxes = segment_digits(image)
    
    if not digit_images:
        # 如果分割失败，尝试作为单个数字处理
        tensor = preprocess_canvas_image(image)
        digit, confidence, probs = predict(model, tensor, device)
        return str(digit), [{
            'digit': digit,
            'confidence': confidence,
            'bbox': None
        }]
    
    # 逐个识别
    result = ""
    details = []
    
    for i, digit_img in enumerate(digit_images):
        tensor = preprocess_single_digit(digit_img)
        digit, confidence, probs = predict(model, tensor, device)
        
        result += str(digit)
        details.append({
            'digit': digit,
            'confidence': confidence,
            'bbox': bboxes[i] if i < len(bboxes) else None
        })
    
    return result, details


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MNIST 数字识别')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='要识别的图像路径')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='模型文件路径（可选）')
    args = parser.parse_args()
    
    # 检查图像文件
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)
    
    # 加载模型
    try:
        model, device = load_model(args.model)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    # 预处理图像
    print(f"\n正在识别图像: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # 预测
    digit, confidence, probs = predict(model, image_tensor, device)
    
    # 输出结果
    print("\n" + "=" * 40)
    print(f"识别结果: {digit}")
    print(f"置信度: {confidence * 100:.2f}%")
    print("=" * 40)
    
    print("\n各数字概率:")
    for i, prob in enumerate(probs):
        bar = "█" * int(prob * 20)
        print(f"  {i}: {prob * 100:5.2f}% {bar}")


if __name__ == "__main__":
    main()
