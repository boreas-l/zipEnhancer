# ------ coding : utf-8 ------
# @FileName     : losses.py
# @Time         : 2025/3/4

"""ZipEnhancer 多损失函数（Multi-Loss）"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1TimeDomainLoss(nn.Module):
    """时域 L1 损失"""

    def forward(self, pred_wav, clean_wav):
        return F.l1_loss(pred_wav, clean_wav)


class L2TimeDomainLoss(nn.Module):
    """时域 L2 (MSE) 损失"""

    def forward(self, pred_wav, clean_wav):
        return F.mse_loss(pred_wav, clean_wav)


class STFTLoss(nn.Module):
    """单分辨率 STFT 损失（频谱收敛损失 + 幅度损失）"""

    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def _stft(self, x):
        # x: [B, T]
        window = self.window.to(x.device)
        spec = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=window, return_complex=True, center=True, pad_mode='reflect'
        )
        mag = spec.abs()
        return mag

    def forward(self, pred_wav, clean_wav):
        pred_mag = self._stft(pred_wav)
        clean_mag = self._stft(clean_wav)

        # 频谱收敛损失
        sc_loss = torch.norm(clean_mag - pred_mag, p="fro") / (torch.norm(clean_mag, p="fro") + 1e-8)
        # 幅度损失
        mag_loss = F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(clean_mag + 1e-7))

        return sc_loss + mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率 STFT 损失"""

    def __init__(self, fft_sizes=(512, 1024, 256), hop_sizes=(128, 256, 64), win_sizes=(512, 1024, 256)):
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(STFTLoss(n_fft, hop, win))

    def forward(self, pred_wav, clean_wav):
        loss = 0.0
        for stft_loss in self.stft_losses:
            loss += stft_loss(pred_wav, clean_wav)
        return loss / len(self.stft_losses)


class CombinedLoss(nn.Module):
    """组合多损失函数"""

    def __init__(self, l1_weight=1.0, l2_weight=1.0, stft_weight=1.0,
                 fft_sizes=(512, 1024, 256), hop_sizes=(128, 256, 64), win_sizes=(512, 1024, 256)):
        super().__init__()
        self.l1_loss = L1TimeDomainLoss()
        self.l2_loss = L2TimeDomainLoss()
        self.stft_loss = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_sizes)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.stft_weight = stft_weight

    def forward(self, pred_wav, clean_wav):
        """
        Args:
            pred_wav: [B, T] 预测的降噪波形
            clean_wav: [B, T] 干净参考波形
        Returns:
            total_loss, loss_dict
        """
        l1 = self.l1_loss(pred_wav, clean_wav)
        l2 = self.l2_loss(pred_wav, clean_wav)
        stft = self.stft_loss(pred_wav, clean_wav)

        total = self.l1_weight * l1 + self.l2_weight * l2 + self.stft_weight * stft

        loss_dict = {
            "l1": l1.item(),
            "l2": l2.item(),
            "stft": stft.item(),
            "total": total.item(),
        }
        return total, loss_dict
