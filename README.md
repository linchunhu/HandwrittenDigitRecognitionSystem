# 🧠 手写数字识别系统

![项目演示](docs/multi_digit_demo_easyocr.png)

基于 **PyTorch** 深度学习框架和 **EasyOCR** 实现的高级手写数字识别系统。提供现代化的 Web 界面，不仅支持单个标准数字识别，还支持**多位数字**、**连续书写**和**交叉笔画**的复杂手写场景。

## ✨ 功能特点

- 🔢 **多位数字识别** - 支持同时识别多个数字（如 "2024"）
- ✍️ **连写支持** - 基于 EasyOCR，无需数字间留有空隙，支持自然书写
- 🎯 **高准确率** - 结合 CNN 模型（99.39%）和 OCR 技术
- ️ **实时交互** - Web 画布支持手写输入，即时返回识别详情
- 🎨 **现代 UI** - 深色主题，响应式设计，动态展示置信度
- 🔧 **混合引擎** - 支持单数字模式（CNN）和多数字模式（OCR）

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装步骤

**1. 克隆项目**

```bash
git clone <repository-url>
cd AI
```

**2. 创建虚拟环境（推荐）**

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

**3. 安装依赖**

```bash
pip install -r requirements.txt
```

**4. 启动 Web 应用**

```bash
python web/app.py
```

> 💡 首次运行时会自动下载 EasyOCR 模型文件（约 100MB），请耐心等待。
> 如果需要训练自己的 CNN 模型，可运行 `python src/train.py`。

**5. 打开浏览器**

访问 [http://localhost:5000](http://localhost:5000)

---

## 📖 使用方法

1. 在左侧画布中随意手写数字（支持多位、连写）
2. 点击「识别」按钮
3. 右侧显示完整的识别结果（如 "785"）和每个数字的置信度
4. 点击「清除」可重新书写

**使用技巧**：
- 支持连续书写，笔画可以交叉
- 识别不准时尝试写大一点
- 首次识别速度稍慢，之后会很快

---

## 📁 项目结构

```
F:\develop\AI\
├── .venv/                   # Python 虚拟环境
├── data/                    # 数据集目录
├── docs/                    # 文档资源
├── models/                  # 模型权重
│   └── mnist_cnn_best.pth   # CNN 模型权重
├── src/                     # 核心源代码
│   ├── model.py             # CNN 模型定义
│   ├── train.py             # 训练脚本
│   ├── predict.py           # CNN 预测脚本
│   └── ocr_engine.py        # EasyOCR 识别引擎 [新增]
├── web/                     # Web 应用
│   ├── app.py               # Flask 后端
│   └── ...
├── requirements.txt         # Python 依赖
└── README.md                # 项目说明
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **识别引擎** | **EasyOCR** (主要), **PyTorch CNN** (辅助) |
| **图像处理** | OpenCV, Pillow, torchvision |
| **Web 后端** | Flask |
| **Web 前端** | HTML5 Canvas, CSS3 (Modern Dark UI), JavaScript |
| **数据集** | MNIST (用于 CNN 训练) |

---

## 📝 更新日志

- **v2.0** - 集成 EasyOCR，支持多位数字连续书写识别
- **v1.2** - 优化分割算法（垂直投影法）
- **v1.1** - 添加智能预处理（自动居中、边界检测）
- **v1.0** - 初始版本，基础 CNN 模型和 Web 界面

---

## 📄 许可证

MIT License
