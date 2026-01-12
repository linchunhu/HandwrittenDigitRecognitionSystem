"""
Flask Web 应用
提供手写数字识别的 Web 界面
"""

import os
import sys
import io
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_engine import recognize_digits, preprocess_for_ocr

app = Flask(__name__)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_digit():
    """接收图像并返回预测结果（使用本地 CNN 模型）"""
    try:
        # 获取图像数据
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'success': False,
                'error': '未接收到图像数据'
            }), 400
        
        # 解码 base64 图像
        # 移除 data URL 前缀
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 RGB（如果有 alpha 通道）
        if image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        # 预处理
        image = preprocess_for_ocr(image)
        
        # 使用本地模型识别
        result, details, avg_confidence = recognize_digits(image)
        
        if not result:
            return jsonify({
                'success': True,
                'result': '',
                'digit': '',
                'confidence': 0,
                'details': [],
                'count': 0,
                'message': '未识别到数字，请写清晰一些'
            })
        
        return jsonify({
            'success': True,
            'result': result,  # 完整识别结果，如 "118"
            'digit': result,
            'confidence': round(avg_confidence * 100, 2),
            'details': [{
                'digit': str(d['digit']), # 确保转为字符串
                'confidence': round(d['confidence'] * 100, 2)
            } for d in details],
            'count': len(details)  # 识别到的文本块个数
        })
    
    except Exception as e:
        print(f"预测错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'engine': 'Local CNN (MNIST)'
    })


if __name__ == '__main__':
    print("=" * 50)
    print("手写数字识别 Web 应用 (Local CNN)")
    print("=" * 50)
    
    print("\n启动服务器...")
    print("请在浏览器中打开: http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
