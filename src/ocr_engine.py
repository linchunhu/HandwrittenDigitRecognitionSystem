"""
本地 OCR 识别模块
使用本地训练的 CNN 模型进行数字识别
"""

import os
import torch
from src.predict import load_model, predict_multi_digits

# 全局模型实例（延迟初始化）
_model = None
_device = None


def get_local_model():
    """获取本地模型（单例模式）"""
    global _model, _device
    if _model is None:
        print("正在加载本地 CNN 模型...")
        try:
            _model, _device = load_model()
            print("本地模型加载完成！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise e
    return _model, _device


def recognize_digits(image):
    """
    使用本地模型识别图像中的数字
    
    Args:
        image: PIL Image 对象（白底黑字）
    
    Returns:
        result: 识别结果字符串（仅包含数字）
        details: 每个检测到的文本的详细信息
        confidence: 平均置信度
    """
    model, device = get_local_model()
    
    # 使用 predict 模块的多数字识别功能
    # 注意：predict_multi_digits 内部会处理图像分割和预处理
    result, details = predict_multi_digits(model, image, device)
    
    # 计算平均置信度
    if details:
        total_conf = sum(d['confidence'] for d in details)
        avg_confidence = total_conf / len(details)
    else:
        avg_confidence = 0.0
        
    return result, details, avg_confidence


def preprocess_for_ocr(image):
    """
    预处理图像
    
    Args:
        image: PIL Image 对象
    
    Returns:
        processed: 处理后的 PIL Image
    """
    # 本地模型在 predict_multi_digits 内部会有专门的预处理
    # 这里只需确保是 RGB 或 L 模式，简单处理即可
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image
