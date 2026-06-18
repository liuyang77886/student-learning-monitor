# 学生学习行为监督和辅导系统

一个基于计算机视觉和人工智能的智能学习监督平台，用于监督孩子写作业时的行为规范，并提供个性化的学习辅导和鼓励。

## 🎯 核心功能

### 行为规范监督
- ✅ **姿态检测**：实时检测低头、弯腰等不良坐姿，温和提醒
- ✅ **注意力追踪**：检测笔长时间停下或分心，立刻介入引导
- ✅ **多模态提醒**：语音 + 视觉反馈，温和提醒学生

### AI 辅导引导
- ✅ **作业题分析**：使用 OCR 识别作业题，智能分析题型
- ✅ **启发式提示**：根据题型给出启发性问题，严禁直接给答案
- ✅ **实时反馈**：监测学生思考过程，给予鼓励性话语
- ✅ **学习进度追踪**：统计学习时长、改进方向、成就记录

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| **视觉检测** | YOLOv8 | 实时物体检测和人员检测 |
| **姿态估计** | MediaPipe | 骨骼关键点检测 |
| **笔迹追踪** | OpenCV + 光流法 | 检测笔的运动和停顿 |
| **AI 辅导** | LLaMA 2 / ChatGLM | 生成启发性提示 |
| **OCR** | PaddleOCR / Tesseract | 识别作业题文本 |
| **移动部署** | ONNX + TensorFlow Lite | 低功耗手机推理 |
| **UI 框架** | Streamlit / Flutter | 跨平台用户界面 |

## 📦 项目结构

```
student-learning-monitor/
├── README.md
├── requirements.txt
├── config.yaml
│
├── core/                           # 核心检测模块
│   ├── __init__.py
│   ├── posture_detector.py        # 姿态检测（低头/弯腰）
│   ├── attention_tracker.py       # 注意力追踪（分心检测）
│   ├── pen_motion_detector.py     # 笔动作检测（停顿检测）
│   └── multi_detector.py          # 多模块协调器
│
├── tutoring/                       # AI 辅导模块
│   ├── __init__.py
│   ├── question_analyzer.py       # 作业题分析和分类
│   ├── hint_generator.py          # 启发式提示生成器
│   ├── encouragement_system.py    # 鼓励反馈系统
│   ├── llm_interface.py           # LLM 接口（支持多种模型）
│   └── prompt_templates.py        # 提示词模板库
│
├── vision/                         # 视觉处理模块
│   ├── __init__.py
│   ├── yolo_detector.py           # YOLOv8 集成
│   ├── pose_estimator.py          # MediaPipe 姿态估计
│   ├── ocr_processor.py           # OCR 文本识别
│   └── frame_processor.py         # 视频帧处理
│
├── utils/                          # 工具函数
│   ├── __init__.py
│   ├── logger.py                  # 日志系统
│   ├── config_manager.py          # 配置管理
│   ├── visualization.py           # 可视化工具
│   ├── audio_feedback.py          # 音频反馈
│   └── data_utils.py              # 数据处理工具
│
├── models/                         # 预训练模型管理
│   ├── __init__.py
│   ├── model_downloader.py        # 模型下载器
│   └── model_converter.py         # 模型格式转换（ONNX/TFLite）
│
├── mobile/                         # 移动端部署
│   ├── __init__.py
│   ├── lite_inference.py          # 轻量级推理引擎
│   ├── android_bridge.py          # Android 接口
│   └── optimization.py            # 模型优化策略
│
├── ui/                             # 用户界面
│   ├── __init__.py
│   ├── app.py                     # Streamlit 主应用
│   ├── dashboard.py               # 学习仪表板
│   ├── settings.py                # 设置页面
│   └── components.py              # UI 组件库
│
├── data/                           # 数据和资源
│   ├── prompts/                   # 启发式提示词库
│   ├── hints/                     # 按题型分类的提示
│   ├── sounds/                    # 提醒音效
│   └── demo_videos/               # 演示视频
│
├── tests/                          # 单元测试
│   ├── __init__.py
│   ├── test_posture_detector.py
│   ├── test_attention_tracker.py
│   ├── test_hint_generator.py
│   └── test_integration.py
│
└── examples/                       # 使用示例
    ├── real_time_monitoring.py    # 实时监控示例
    ├── batch_analysis.py          # 批处理分析
    └── mobile_deployment.py       # 移动端部署示例
```

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/liuyang77886/student-learning-monitor.git
cd student-learning-monitor

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

```python
from core.multi_detector import MonitoringSystem
from tutoring.hint_generator import HintGenerator
import cv2

# 初始化监督系统
monitor = MonitoringSystem(config_path="config.yaml")

# 初始化辅导系统
tutor = HintGenerator(model_name="ChatGLM")

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 检测姿态和注意力
    results = monitor.analyze_frame(frame)
    
    # 触发提醒或辅导
    if results['bad_posture']:
        monitor.send_reminder("请保持正确的坐姿")
    
    if results['distracted']:
        tutor.generate_encouragement("我发现你分心了，让我们重新回到题目")
    
    # 显示结果
    annotated_frame = monitor.visualize(frame, results)
    cv2.imshow('Student Learning Monitor', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 主要模块说明

### 1. 姿态检测模块 (`posture_detector.py`)
- **功能**：实时检测学生的坐姿
- **检测指标**：
  - 头部相对肩膀的位置（检测低头）
  - 脊椎弯曲度（检测弯腰）
  - 头部倾斜角度
- **输出**：姿态状态、置信度、改正建议

### 2. 注意力追踪模块 (`attention_tracker.py`)
- **功能**：检测学生是否分心
- **追踪指标**：
  - 眼睛/头部移动频率和幅度
  - 笔的运动频率
  - 面部表情变化
- **阈值**：连续 3 秒无笔动作 = 分心状态

### 3. 笔动作检测模块 (`pen_motion_detector.py`)
- **功能**：检测笔的停顿时间
- **检测方法**：
  - 光流法追踪笔的运动
  - 手部关键点检测
  - 动作连贯性分析
- **输出**：笔停顿时长、运动轨迹

### 4. 作业题分析模块 (`question_analyzer.py`)
- **功能**：识别和分类作业题
- **支持题型**：
  - 选择题、填空题、解答题
  - 数学、语文、英语等科目
  - 单道题、多道题混合
- **输出**：题型分类、难度评估、推荐提示

### 5. 启发式提示生成 (`hint_generator.py`)
- **功能**：根据题型生成启发性提示
- **提示策略**：
  - 分解问题：将复杂问题分解为小步骤
  - 逆向启发：从答案反推解题思路
  - 类比启发：给出相似的已知问题
  - 概念复习：复习相关的基础概念
- **严格约束**：禁止直接给出答案

### 6. 鼓励反馈系统 (`encouragement_system.py`)
- **功能**：在学习过程中给予实时鼓励
- **反馈类型**：
  - 进度鼓励："你已经做了 3 道题，继续加油！"
  - 方法表扬："你的思路很清晰！"
  - 困难鼓励："这道题很难，但你坚持思考，很棒！"
  - 成长反馈："你的分析能力在进步！"

## 🎨 可视化界面

### Streamlit 仪表板
- 📹 实时视频监控窗口
- 📊 行为统计图表
- ⚠️ 实时警告和提醒
- 🎯 学习进度追踪
- 💬 对话记录

## 📱 移动端部署

### 模型优化
- YOLO 模型量化（INT8）
- MediaPipe Lite 版本
- TensorFlow Lite 转换
- ONNX Runtime 移动部署

### 支持的平台
- ✅ Android（TensorFlow Lite）
- ✅ iOS（Core ML）
- ✅ Web（ONNX.js）

## 🔧 配置文件示例

```yaml
# config.yaml
monitoring:
  posture:
    bad_posture_threshold: 0.7
    reminder_cooldown: 30  # 秒
    alert_type: "voice+visual"
  
  attention:
    distraction_timeout: 3  # 秒
    min_pen_movement: 0.5   # 像素/帧
  
tutoring:
  model: "ChatGLM"  # or "LLaMA", "GPT"
  hint_level: "medium"  # easy, medium, hard
  encouragement_interval: 60  # 秒
  
vision:
  yolo_model: "yolov8n"
  pose_model: "mediapipe"
  ocr_model: "paddleocr"
  
mobile:
  enable_quantization: true
  target_fps: 15
  model_format: "tflite"
```

## 📈 性能指标

| 指标 | 目标 | 状态 |
|------|------|------|
| 实时处理速度 | 20+ FPS（手机）| 🚧 开发中 |
| 姿态检测准确率 | > 95% | 🚧 测试中 |
| 注意力检测延迟 | < 500ms | 🚧 优化中 |
| 提示生成速度 | < 2s | 🚧 集成中 |
| 内存占用 | < 500MB（手机）| 🚧 测试中 |

## 📚 核心依赖

- `ultralytics`（YOLOv8）
- `mediapipe`（姿态估计）
- `opencv-python`（图像处理）
- `paddleocr` 或 `pytesseract`（文本识别）
- `transformers`（LLM 接口）
- `streamlit`（UI）
- `numpy`, `scipy`（数据处理）

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature')`
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

## 📝 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 💡 后续计划

- [ ] 支持多语言（中文、英文、日文等）
- [ ] 云端数据同步
- [ ] 家长实时监控面板
- [ ] 学习报告生成
- [ ] 与学校系统对接
- [ ] 增强现实（AR）辅导
- [ ] 多模态学习分析

## 📞 联系方式

- 📧 Email: your.email@example.com
- 🐙 GitHub Issues: [提交问题](https://github.com/liuyang77886/student-learning-monitor/issues)
- 💬 讨论: [参与讨论](https://github.com/liuyang77886/student-learning-monitor/discussions)

---

**特别感谢：** 感谢 Google MediaPipe、Ultralytics YOLOv8 和开源社区的贡献！

⭐ 如果这个项目对您有帮助，请给个 Star！