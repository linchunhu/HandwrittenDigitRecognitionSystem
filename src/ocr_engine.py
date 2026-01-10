"""
EasyOCR 识别模块
使用 EasyOCR 实现更强大的多数字识别
"""

import io
import easyocr
import numpy as np
from PIL import Image

# 全局 EasyOCR reader（延迟初始化以节省启动时间）
_reader = None


def get_reader():
    """获取 EasyOCR reader（单例模式）"""
    global _reader
    if _reader is None:
        print("正在初始化 EasyOCR（首次加载需要下载模型）...")
        # 只识别数字，使用英文模型（更快）
        _reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR 初始化完成！")
    return _reader


def recognize_digits_easyocr(image):
    """
    使用 EasyOCR 识别图像中的数字
    
    Args:
        image: PIL Image 对象（白底黑字）
    
    Returns:
        result: 识别结果字符串（仅包含数字）
        details: 每个检测到的文本的详细信息
        confidence: 平均置信度
    """
    reader = get_reader()
    
    # 转换为 numpy 数组
    img_array = np.array(image.convert('RGB'))
    
    # 使用 EasyOCR 识别
    # 调整参数以提高手写数字识别效果
    results = reader.readtext(
        img_array,
        allowlist='0123456789',
        paragraph=False,
        min_size=5,           # 降低最小尺寸
        text_threshold=0.3,   # 降低文字检测阈值
        low_text=0.2,         # 降低低置信度阈值
        link_threshold=0.2,   # 降低链接阈值
        canvas_size=1280,     # 放大画布
        mag_ratio=2.0         # 放大倍率
    )
    
    if not results:
        return "", [], 0
    
    # 按 x 坐标排序（从左到右）
    results.sort(key=lambda x: x[0][0][0])
    
    # 提取结果
    recognized_text = ""
    details = []
    total_confidence = 0
    
    for bbox, text, confidence in results:
        # 只保留数字
        digits = ''.join(c for c in text if c.isdigit())
        if digits:
            recognized_text += digits
            details.append({
                'text': digits,
                'confidence': confidence,
                'bbox': bbox
            })
            total_confidence += confidence
    
    avg_confidence = total_confidence / len(details) if details else 0
    
    return recognized_text, details, avg_confidence


def preprocess_for_ocr(image):
    """
    为 OCR 预处理图像
    
    Args:
        image: PIL Image 对象
    
    Returns:
        processed: 处理后的 PIL Image
    """
    from PIL import ImageOps, ImageFilter
    
    # 确保是 RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 轻微锐化以增强边缘
    image = image.filter(ImageFilter.SHARPEN)
    
    return image
