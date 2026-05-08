# ZipEnhancer
- 【来源】该项目来源于阿里通义实验室开源的语音降噪模型ZipEnhancer >>> [项目连接](https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base/summary)  
- 【背景】最近因为项目问题，打算调研测试一下最新的通话语音降噪算法，看了一下，发现阿里开源的ZipEnhancer模型效果好像还不错，打算试一试。 因为modelscope封装比较复杂，调用链路比较深，为了方便修改与测试，基于pytorch，将ZipEnhancer模型的代码逻辑剥离了出来。
- 【记录】
    - 2025.3.4：简单把推理部分提取了出来，并完成测试，基于官方测试用例的去噪效果还不错。
    - 接下来计划更新onnx导出部分，并完善训练部分代码。【官方目前还没提供训练部分代码】
    - 2026.05.09: 补充onnx导出部分以及训练部分

## 项目结构

```
zipEnhancer/
├── inference_pipeline.py       # 推理入口
├── export_onnx.py              # ONNX 导出脚本
├── train.py                    # 训练脚本
├── dataset.py                  # 训练数据集
├── losses.py                   # 多损失函数
├── utils.py                    # 工具函数
├── configs/
│   ├── configuration.json      # 模型配置
│   └── train_config.json       # 训练配置
├── models/
│   ├── zipenhancer.py          # 核心模型
│   └── layers/
│       ├── generator.py        # 编码器/解码器
│       ├── zipenhancer_layer.py# 双路径编码器
│       ├── zipformer.py        # Zipformer2 核心层
│       └── scaling.py          # 缩放/归一化工具
├── pytorch_model.bin           # 预训练权重
└── test_datas/                 # 测试音频
```

## 环境依赖

```bash
pip install torch librosa soundfile numpy
# ONNX 导出额外依赖
pip install onnx onnxruntime onnxscript
# 训练额外依赖（可选）
pip install tensorboard
```

## 推理

```bash
python inference_pipeline.py
```

输入含噪 WAV 文件，输出降噪后的 PCM 音频。支持任意长度音频（长音频自动分段处理）。

## ONNX 导出

将 PyTorch 模型导出为 ONNX 格式，支持动态长度输入，便于部署到各类推理引擎。

### 使用方式

```bash
# 基础导出
python export_onnx.py

# 导出并验证精度（对比 PyTorch 与 ONNX Runtime 输出）
python export_onnx.py --verify

# 自定义参数
python export_onnx.py --weight pytorch_model.bin --config ./configs/configuration.json --output zipenhancer.onnx --opset 17 --verify
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--weight` | `pytorch_model.bin` | 模型权重路径 |
| `--config` | `./configs/configuration.json` | 模型配置文件 |
| `--output` | `zipenhancer.onnx` | ONNX 输出路径 |
| `--opset` | `17` | ONNX opset 版本（需 >= 17） |
| `--verify` | - | 是否验证导出精度 |

### 技术细节

- 使用 Conv1d 实现 ONNX 兼容的 STFT（替代 `torch.stft` 的复数输出）
- 使用 ConvTranspose1d 实现 ONNX 兼容的 iSTFT overlap-add
- 支持 `dynamic_axes`：batch 和 time 维度均为动态，可处理任意长度音频
- 验证标准：PyTorch 与 ONNX Runtime 输出的 Cosine Similarity > 0.99

## 训练

基于多损失函数（Multi-Loss）的完整训练框架，支持从预训练权重微调或从头训练。

### 数据准备

训练数据需组织为以下目录结构：

**模式1 - 配对模式（paired）**：noisy 和 clean 文件一一对应
```
data/
├── train/
│   ├── noisy/    # 含噪语音
│   │   ├── 001.wav
│   │   └── 002.wav
│   └── clean/    # 干净语音
│       ├── 001.wav
│       └── 002.wav
└── val/
    ├── noisy/
    └── clean/
```

**模式2 - 动态混合模式（mix）**：自动按随机 SNR 混合 clean + noise
```
data/
├── train/
│   ├── clean/    # 干净语音
│   └── noise/    # 噪声文件
└── val/
    ├── clean/
    └── noise/
```

### 使用方式

```bash
# 单卡训练
python train.py --config configs/train_config.json

# 多卡分布式训练（DDP）
torchrun --nproc_per_node=4 train.py --config configs/train_config.json --distributed
```

### 训练配置 (`configs/train_config.json`)

| 配置项 | 说明 |
|--------|------|
| `data.train_dir` / `data.val_dir` | 训练/验证数据目录 |
| `data.mode` | 数据模式：`paired`（配对）或 `mix`（动态混合） |
| `data.segment_length` | 训练片段长度（采样点），默认 32000（4秒） |
| `data.snr_range` | 动态混合 SNR 范围 (dB)，默认 [-5, 20] |
| `training.batch_size` | 批大小，默认 8 |
| `training.epochs` | 训练轮数，默认 100 |
| `optimizer.lr` | 学习率，默认 1e-3 |
| `scheduler.warmup_epochs` | Warmup 轮数，默认 5 |
| `loss.l1_weight / l2_weight / stft_weight` | 各损失权重 |
| `checkpoint.resume` | 恢复训练的 checkpoint 路径（null 表示不恢复） |
| `checkpoint.pretrained` | 预训练权重路径（默认 `pytorch_model.bin`，设为 null 则从头训练） |

### 训练模式

支持三种初始化方式（按优先级）：

1. **断点续训**：设置 `checkpoint.resume` 为之前保存的 checkpoint 路径
2. **预训练微调**（默认）：使用 `checkpoint.pretrained` 指定的预训练权重初始化
3. **从头训练**：将 `checkpoint.pretrained` 设为 `null`，模型随机初始化

### 损失函数

采用多损失联合训练（与模型名 `multiloss` 对应）：

- **L1 时域损失**：预测波形与干净波形的 L1 距离
- **L2 时域损失**：预测波形与干净波形的 MSE
- **Multi-Resolution STFT Loss**：多分辨率频谱损失（频谱收敛 + 对数幅度），使用 3 种 FFT 配置 (512/1024/256)

### 训练特性

- **学习率调度**：Warmup + Cosine Decay
- **梯度裁剪**：max_norm=5.0，防止梯度爆炸
- **数据增强**：随机增益、动态 SNR 混合
- **评估指标**：SI-SNR (dB)
- **分布式训练**：支持 PyTorch DDP 多卡并行
- **TensorBoard 日志**：实时监控 loss 和学习率曲线
- **Checkpoint**：定期保存，支持断点续训
