# ------ coding : utf-8 ------
# @FileName     : train.py
# @Time         : 2025/3/4

"""
ZipEnhancer 完整训练脚本
支持: DDP多卡训练、TensorBoard日志、学习率调度、Checkpoint恢复

用法:
    单卡: python train.py --config configs/train_config.json
    多卡: torchrun --nproc_per_node=4 train.py --config configs/train_config.json --distributed
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.zipenhancer import ZipenhancerDecorator, mag_pha_stft, mag_pha_istft
from dataset import SpeechEnhancementDataset
from losses import CombinedLoss
from utils import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_epochs = cfg["scheduler"]["warmup_epochs"]
    total_epochs = cfg["training"]["epochs"]
    min_lr = cfg["scheduler"]["min_lr"]
    base_lr = cfg["optimizer"]["lr"]
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_lr / base_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def si_snr(pred, target):
    """Scale-Invariant SNR"""
    target = target - target.mean(dim=-1, keepdim=True)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    dot = torch.sum(target * pred, dim=-1, keepdim=True)
    s_target = dot * target / (torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8)
    noise = pred - s_target
    si_snr_val = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
    )
    return si_snr_val.mean()


class Trainer:

    def __init__(self, cfg, local_rank=0, world_size=1):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        set_seed(cfg["training"]["seed"] + local_rank)

        # 模型
        model_cfg_path = "./configs/configuration.json"
        model_cfg, _ = Config.file2dict(model_cfg_path)
        model_configs = model_cfg.get('model')

        weight_path = cfg["checkpoint"].get("resume")
        pretrained_path = cfg["checkpoint"].get("pretrained", "pytorch_model.bin")

        if weight_path and os.path.exists(weight_path):
            self.decorator = ZipenhancerDecorator(weight_path, **model_configs)
            print(f"[Rank {local_rank}] 从 checkpoint 恢复: {weight_path}")
        elif pretrained_path and os.path.exists(pretrained_path):
            self.decorator = ZipenhancerDecorator(pretrained_path, **model_configs)
            print(f"[Rank {local_rank}] 使用预训练权重初始化: {pretrained_path}")
        else:
            self.decorator = ZipenhancerDecorator("", **model_configs)
            print(f"[Rank {local_rank}] 无预训练权重，从头训练（随机初始化）")

        self.model = self.decorator.model.to(self.device)
        self.model.train()

        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)

        # 损失函数
        loss_cfg = cfg["loss"]
        self.criterion = CombinedLoss(
            l1_weight=loss_cfg["l1_weight"],
            l2_weight=loss_cfg["l2_weight"],
            stft_weight=loss_cfg["stft_weight"],
            fft_sizes=loss_cfg["fft_sizes"],
            hop_sizes=loss_cfg["hop_sizes"],
            win_sizes=loss_cfg["win_sizes"],
        ).to(self.device)

        # 优化器
        opt_cfg = cfg["optimizer"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
        )

        # 数据集
        data_cfg = cfg["data"]
        self.train_dataset = SpeechEnhancementDataset(
            data_dir=data_cfg["train_dir"],
            sample_rate=data_cfg["sample_rate"],
            segment_length=data_cfg["segment_length"],
            mode=data_cfg["mode"],
            snr_range=data_cfg["snr_range"],
            augment=data_cfg["augment"],
        )
        self.val_dataset = SpeechEnhancementDataset(
            data_dir=data_cfg["val_dir"],
            sample_rate=data_cfg["sample_rate"],
            segment_length=data_cfg["segment_length"],
            mode=data_cfg["mode"],
            snr_range=data_cfg["snr_range"],
            augment=False,
        )

        train_sampler = DistributedSampler(self.train_dataset) if world_size > 1 else None
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            pin_memory=True,
        )
        self.train_sampler = train_sampler

        # 学习率调度
        self.scheduler = get_lr_scheduler(self.optimizer, cfg, len(self.train_loader))

        # TensorBoard
        self.writer = None
        if self.is_main:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = cfg["tensorboard"]["log_dir"]
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(tb_dir)
            except ImportError:
                print(">>> TensorBoard 不可用，跳过日志记录")

        # Checkpoint 目录
        self.save_dir = cfg["checkpoint"]["save_dir"]
        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)

        self.n_fft = 400
        self.hop_size = 100
        self.win_size = 400
        self.compress_factor = 0.3

        self.global_step = 0
        self.start_epoch = 0

    def _forward(self, noisy_wav):
        """模型前向：wav -> STFT -> model -> iSTFT -> wav"""
        norm_factor = torch.sqrt(
            noisy_wav.shape[1] / (torch.sum(noisy_wav ** 2.0, dim=1, keepdim=True) + 1e-8)
        )
        noisy_audio = noisy_wav * norm_factor

        mag, pha, com = mag_pha_stft(
            noisy_audio, self.n_fft, self.hop_size, self.win_size,
            compress_factor=self.compress_factor, center=True
        )

        model = self.model.module if isinstance(self.model, DDP) else self.model
        amp_g, pha_g, com_g, _, _ = model(mag, pha)

        wav = mag_pha_istft(
            amp_g, pha_g, self.n_fft, self.hop_size, self.win_size,
            compress_factor=self.compress_factor, center=True
        )
        wav = wav / norm_factor
        return wav

    def train_epoch(self, epoch):
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        log_interval = self.cfg["training"]["log_interval"]

        for batch_idx, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            pred_wav = self._forward(noisy)

            # 对齐长度
            min_len = min(pred_wav.shape[1], clean.shape[1])
            pred_wav = pred_wav[:, :min_len]
            clean = clean[:, :min_len]

            loss, loss_dict = self.criterion(pred_wav, clean)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss_dict["total"]
            self.global_step += 1

            if self.is_main and batch_idx % log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  [Epoch {epoch}][{batch_idx}/{len(self.train_loader)}] "
                      f"loss={loss_dict['total']:.4f} "
                      f"(l1={loss_dict['l1']:.4f} l2={loss_dict['l2']:.4f} stft={loss_dict['stft']:.4f}) "
                      f"lr={lr:.2e}")
                if self.writer:
                    self.writer.add_scalar("train/loss_total", loss_dict["total"], self.global_step)
                    self.writer.add_scalar("train/loss_l1", loss_dict["l1"], self.global_step)
                    self.writer.add_scalar("train/loss_l2", loss_dict["l2"], self.global_step)
                    self.writer.add_scalar("train/loss_stft", loss_dict["stft"], self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        return epoch_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_si_snr = 0.0
        count = 0

        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)

            pred_wav = self._forward(noisy)

            min_len = min(pred_wav.shape[1], clean.shape[1])
            pred_wav = pred_wav[:, :min_len]
            clean = clean[:, :min_len]

            loss, _ = self.criterion(pred_wav, clean)
            total_loss += loss.item()
            total_si_snr += si_snr(pred_wav, clean).item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        avg_si_snr = total_si_snr / max(count, 1)

        if self.is_main:
            print(f"  [Val Epoch {epoch}] loss={avg_loss:.4f} SI-SNR={avg_si_snr:.2f} dB")
            if self.writer:
                self.writer.add_scalar("val/loss", avg_loss, epoch)
                self.writer.add_scalar("val/si_snr", avg_si_snr, epoch)

        return avg_loss

    def save_checkpoint(self, epoch):
        if not self.is_main:
            return
        model = self.model.module if isinstance(self.model, DDP) else self.model
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(state, path)
        print(f"  [Checkpoint] 已保存: {path}")

    def train(self):
        epochs = self.cfg["training"]["epochs"]
        save_interval = self.cfg["training"]["save_interval"]
        val_interval = self.cfg["training"]["val_interval"]

        if self.is_main:
            print(f">>> 开始训练: {epochs} epochs, {len(self.train_loader)} steps/epoch")
            print(f"    设备: {self.device}, 进程数: {self.world_size}")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            elapsed = time.time() - t0

            if self.is_main:
                print(f">>> Epoch {epoch} 完成: avg_loss={train_loss:.4f}, 耗时={elapsed:.1f}s")

            if (epoch + 1) % val_interval == 0:
                self.validate(epoch)

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch)

        # 保存最终模型
        self.save_checkpoint(epochs - 1)
        if self.is_main and self.writer:
            self.writer.close()
        if self.is_main:
            print(">>> 训练完成!")


def main():
    parser = argparse.ArgumentParser(description="ZipEnhancer Training")
    parser.add_argument("--config", type=str, default="./configs/train_config.json")
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    cfg, _ = Config.file2dict(args.config)

    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        cfg["distributed"]["enabled"] = True
    else:
        local_rank = 0
        world_size = 1

    trainer = Trainer(cfg, local_rank=local_rank, world_size=world_size)
    trainer.train()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
