# ------ coding : utf-8 ------
# @FileName     : export_onnx.py
# @Time         : 2025/3/4

"""ZipEnhancer ONNX 导出脚本，支持动态长度输入"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.zipenhancer import ZipenhancerDecorator
from utils import Config


def onnx_stft(y, n_fft, hop_size, win_size, compress_factor=0.3, center=True):
    """ONNX 兼容的 STFT 实现（使用 Conv1d 代替 torch.stft）"""
    if center:
        pad_amount = n_fft // 2
        y = F.pad(y, (pad_amount, pad_amount), mode='reflect')

    hann_window = torch.hann_window(win_size, device=y.device)

    # 使用 Conv1d 做分帧+DFT（ONNX 完全兼容）
    n_freq = n_fft // 2 + 1
    freq_idx = torch.arange(n_freq, device=y.device, dtype=torch.float32)
    time_idx = torch.arange(n_fft, device=y.device, dtype=torch.float32)
    angle = -2.0 * torch.pi * freq_idx.unsqueeze(1) * time_idx.unsqueeze(0) / n_fft

    # DFT 基函数 * 窗函数 -> Conv1d 权重
    cos_kernel = (torch.cos(angle) * hann_window).unsqueeze(1)  # [n_freq, 1, n_fft]
    sin_kernel = (torch.sin(angle) * hann_window).unsqueeze(1)  # [n_freq, 1, n_fft]

    # y: [B, T] -> [B, 1, T]
    y_unsqueeze = y.unsqueeze(1)
    real = F.conv1d(y_unsqueeze, cos_kernel, stride=hop_size)  # [B, n_freq, num_frames]
    imag = F.conv1d(y_unsqueeze, sin_kernel, stride=hop_size)  # [B, n_freq, num_frames]

    mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-9)
    pha = torch.atan2(imag, real + 1e-5)

    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def onnx_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=0.3, center=True):
    """ONNX 兼容的 iSTFT 实现（使用 ConvTranspose1d 做 overlap-add）"""
    mag = torch.pow(mag, 1.0 / compress_factor)
    real = mag * torch.cos(pha)  # [B, n_freq, num_frames]
    imag = mag * torch.sin(pha)  # [B, n_freq, num_frames]

    n_freq = n_fft // 2 + 1
    hann_window = torch.hann_window(win_size, device=real.device)

    # iDFT 基函数
    freq_idx = torch.arange(n_freq, device=real.device, dtype=torch.float32)
    time_idx = torch.arange(n_fft, device=real.device, dtype=torch.float32)
    angle = 2.0 * torch.pi * freq_idx.unsqueeze(0) * time_idx.unsqueeze(1) / n_fft
    cos_basis_inv = torch.cos(angle) * (2.0 / n_fft)  # [n_fft, n_freq]
    sin_basis_inv = torch.sin(angle) * (2.0 / n_fft)  # [n_fft, n_freq]

    # [B, num_frames, n_freq] x [n_freq, n_fft] -> [B, num_frames, n_fft]
    real_t = real.permute(0, 2, 1)
    imag_t = imag.permute(0, 2, 1)
    frames = torch.matmul(real_t, cos_basis_inv.t()) - torch.matmul(imag_t, sin_basis_inv.t())
    frames = frames * hann_window  # [B, num_frames, n_fft]

    # ConvTranspose1d overlap-add
    frames_t = frames.permute(0, 2, 1)  # [B, n_fft, num_frames]
    num_frames = frames_t.shape[2]

    kernel = torch.eye(n_fft, device=real.device).unsqueeze(1)  # [n_fft, 1, n_fft]
    wav = F.conv_transpose1d(frames_t, kernel, stride=hop_size, groups=n_fft)
    wav = wav.sum(dim=1)  # [B, output_length]

    # 窗函数归一化
    window_sq_frame = (hann_window ** 2).unsqueeze(0).unsqueeze(2).expand(1, n_fft, num_frames)
    window_sum = F.conv_transpose1d(window_sq_frame, kernel, stride=hop_size, groups=n_fft)
    window_sum = window_sum.sum(dim=1)  # [1, output_length]
    window_sum = torch.clamp(window_sum, min=1e-8)

    wav = wav / window_sum

    if center:
        pad_amount = n_fft // 2
        wav = wav[:, pad_amount:-pad_amount]

    return wav


class ZipEnhancerONNX(nn.Module):
    """将 ZipenhancerDecorator 的前向逻辑封装为标准 nn.Module，便于 ONNX 导出"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_fft = 400
        self.hop_size = 100
        self.win_size = 400
        self.compress_factor = 0.3

    def forward(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_wav: [B, T] 原始含噪波形
        Returns:
            wav: [B, T] 降噪后波形
        """
        norm_factor = torch.sqrt(
            noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0, dim=1, keepdim=True)
        )
        noisy_audio = noisy_wav * norm_factor

        mag, pha, com = onnx_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size,
            compress_factor=self.compress_factor, center=True
        )

        amp_g, pha_g, com_g, _, others = self.model(mag, pha)

        wav = onnx_istft(
            amp_g, pha_g, self.n_fft, self.hop_size, self.win_size,
            compress_factor=self.compress_factor, center=True
        )

        wav = wav / norm_factor
        return wav


def export_onnx(weight_path, config_path, output_path="zipenhancer.onnx", opset_version=17):
    """导出 ONNX 模型"""
    cfg_dict, _ = Config.file2dict(config_path)
    model_configs = cfg_dict.get('model')

    decorator = ZipenhancerDecorator(weight_path, **model_configs)
    decorator.model.eval()

    onnx_model = ZipEnhancerONNX(decorator.model)
    onnx_model.eval()

    # 使用 4 秒音频作为示例输入 (8000Hz * 4s = 32000 samples)
    dummy_input = torch.randn(1, 32000)

    print(f">>> 开始导出 ONNX 模型...")
    print(f"    输入形状: {dummy_input.shape}")

    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "output": {0: "batch", 1: "time"},
        },
        dynamo=False,
    )
    print(f">>> ONNX 模型已导出到: {output_path}")
    return onnx_model, dummy_input


def verify_onnx(onnx_model, dummy_input, output_path="zipenhancer.onnx"):
    """验证 ONNX 模型与 PyTorch 模型输出一致性"""
    try:
        import onnxruntime as ort
    except ImportError:
        print(">>> 跳过验证: 请安装 onnxruntime (pip install onnxruntime)")
        return

    # PyTorch 推理
    with torch.no_grad():
        pt_output = onnx_model(dummy_input).numpy()

    # ONNX 推理
    sess = ort.InferenceSession(output_path)
    ort_output = sess.run(None, {"input": dummy_input.numpy()})[0]

    # 对比结果
    cosine_sim = np.dot(pt_output.flatten(), ort_output.flatten()) / (
        np.linalg.norm(pt_output.flatten()) * np.linalg.norm(ort_output.flatten()) + 1e-8
    )
    max_diff = np.max(np.abs(pt_output - ort_output))

    print(f">>> 验证结果:")
    print(f"    Cosine Similarity: {cosine_sim:.6f}")
    print(f"    Max Absolute Diff: {max_diff:.6e}")
    print(f"    PyTorch output shape: {pt_output.shape}")
    print(f"    ONNX output shape: {ort_output.shape}")

    if cosine_sim > 0.99:
        print(">>> ✓ ONNX 导出验证通过!")
    else:
        print(">>> ✗ 警告: 精度差异较大，请检查模型")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="导出 ZipEnhancer ONNX 模型")
    parser.add_argument("--weight", type=str, default="pytorch_model.bin", help="模型权重路径")
    parser.add_argument("--config", type=str, default="./configs/configuration.json", help="配置文件路径")
    parser.add_argument("--output", type=str, default="zipenhancer.onnx", help="ONNX 输出路径")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="是否验证导出结果")
    args = parser.parse_args()

    onnx_model, dummy_input = export_onnx(args.weight, args.config, args.output, args.opset)

    if args.verify:
        verify_onnx(onnx_model, dummy_input, args.output)
