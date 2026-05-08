# ------ coding : utf-8 ------
# @FileName     : dataset.py
# @Time         : 2025/3/4

"""语音降噪数据集：支持 noisy/clean 配对和动态混合增强"""

import os
import random

import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset

from utils import audio_norm


class SpeechEnhancementDataset(Dataset):
    """
    语音降噪训练数据集

    目录结构（模式1 - 配对模式）:
        data_dir/
            noisy/   # 含噪语音
                001.wav
                002.wav
            clean/   # 干净语音
                001.wav
                002.wav

    目录结构（模式2 - 动态混合模式）:
        data_dir/
            clean/   # 干净语音
            noise/   # 噪声文件
    """

    def __init__(self, data_dir, sample_rate=8000, segment_length=32000,
                 mode="paired", snr_range=(-5, 20), augment=True):
        """
        Args:
            data_dir: 数据根目录
            sample_rate: 目标采样率
            segment_length: 训练片段长度（采样点数）
            mode: "paired"(配对) 或 "mix"(动态混合)
            snr_range: 动态混合时的 SNR 范围 (dB)
            augment: 是否启用数据增强
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.mode = mode
        self.snr_range = snr_range
        self.augment = augment

        if mode == "paired":
            self.noisy_dir = os.path.join(data_dir, "noisy")
            self.clean_dir = os.path.join(data_dir, "clean")
            self.file_list = sorted([
                f for f in os.listdir(self.clean_dir)
                if f.endswith(('.wav', '.flac', '.mp3'))
            ])
        elif mode == "mix":
            self.clean_dir = os.path.join(data_dir, "clean")
            self.noise_dir = os.path.join(data_dir, "noise")
            self.file_list = sorted([
                f for f in os.listdir(self.clean_dir)
                if f.endswith(('.wav', '.flac', '.mp3'))
            ])
            self.noise_list = sorted([
                f for f in os.listdir(self.noise_dir)
                if f.endswith(('.wav', '.flac', '.mp3'))
            ])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def __len__(self):
        return len(self.file_list)

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        return audio.astype(np.float32)

    def _random_crop(self, audio, length):
        if len(audio) >= length:
            start = random.randint(0, len(audio) - length)
            return audio[start:start + length]
        else:
            return np.pad(audio, (0, length - len(audio)), mode='constant')

    def _mix_snr(self, clean, noise, snr_db):
        """按指定 SNR 混合"""
        clean_power = np.sum(clean ** 2) + 1e-8
        noise_power = np.sum(noise ** 2) + 1e-8
        scale = np.sqrt(clean_power / (noise_power * (10 ** (snr_db / 10))))
        noisy = clean + scale * noise
        return noisy

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        if self.mode == "paired":
            clean_path = os.path.join(self.clean_dir, filename)
            noisy_path = os.path.join(self.noisy_dir, filename)
            clean = self._load_audio(clean_path)
            noisy = self._load_audio(noisy_path)

            clean = self._random_crop(clean, self.segment_length)
            noisy = self._random_crop(noisy, self.segment_length)

        elif self.mode == "mix":
            clean_path = os.path.join(self.clean_dir, filename)
            clean = self._load_audio(clean_path)
            clean = self._random_crop(clean, self.segment_length)

            noise_file = random.choice(self.noise_list)
            noise = self._load_audio(os.path.join(self.noise_dir, noise_file))
            noise = self._random_crop(noise, self.segment_length)

            snr_db = random.uniform(*self.snr_range)
            noisy = self._mix_snr(clean, noise, snr_db)

        if self.augment:
            # 随机增益
            gain = random.uniform(0.5, 1.5)
            clean = clean * gain
            noisy = noisy * gain

        clean = audio_norm(clean)
        noisy = audio_norm(noisy)

        return (
            torch.from_numpy(noisy).float(),
            torch.from_numpy(clean).float(),
        )
