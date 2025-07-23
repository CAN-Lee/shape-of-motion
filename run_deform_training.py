#!/usr/bin/env python3
"""
Gaussian Deformation Training Script

This script trains a deformation model for Gaussian sequences using temporal graph networks.

Key Features:
- Temporal LSTM-based deformation model
- Graph Neural Network for spatial refinement
- Gaussian sequence processing with optional chunked processing
- L1 + SSIM combined loss function for better image quality
- Complete scene rendering (foreground + background) for accurate loss computation

Memory Optimization Options:
- use_chunked_processing: If True, processes Gaussians in chunks to save memory (default: True)
- chunk_size: Size of each processing chunk (default: 1000)
- Setting use_chunked_processing=False processes all Gaussians at once (faster but uses more memory)

Usage:
    python run_deform_training.py --work_dir path/to/work --scene_model_path path/to/model.ckpt [data_config] [options]

Examples:
    # Use chunked processing (memory efficient)
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --use_chunked_processing true --chunk_size 500 nvidia

    # Use full processing (faster but more memory)
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --use_chunked_processing false nvidia

    # Resume training from latest checkpoint (automatically detected)
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt nvidia

    # Resume training from specific checkpoint
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --resume_from ./output/deform_checkpoint_epoch_50.pth nvidia

    # Render full sequence for better temporal consistency
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --render_full_sequence true nvidia

    # Render only center frame (faster training)
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --render_full_sequence false nvidia

    # Adjust SSIM loss weight (default: 0.2)
    python run_deform_training.py --work_dir ./output --scene_model_path ./model.ckpt --ssim_weight 0.1 nvidia
"""

import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data import (
    BaseDataset,
    DavisDataConfig,
    CustomDataConfig,
    get_train_val_datasets,
    iPhoneDataConfig,
    NvidiaDataConfig,
)
from flow3d.data.utils import to_device
from flow3d.scene_model import SceneModel
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server

from Gaussian_Sequence.LSTM_Graph_Refinement import TemporalGraphRefinement, GaussianRefiner
from Gaussian_Sequence.rgb_loss_utils import l1_loss, ssim
import pdb

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class DeformTrainConfig:
    # 必须的参数（没有默认值）
    data: (
        Annotated[iPhoneDataConfig, tyro.conf.subcommand(name="iphone")]
        | Annotated[DavisDataConfig, tyro.conf.subcommand(name="davis")]
        | Annotated[CustomDataConfig, tyro.conf.subcommand(name="custom")]
        | Annotated[NvidiaDataConfig, tyro.conf.subcommand(name="nvidia")]
    )
    # 可选参数（有默认值）
    work_dir: str = "refined_output"
    scene_model_path: str = "output/paper-windmill-result/checkpoints/last.ckpt"  # 预训练的scene model路径
    resume_from: str | None = None  # 指定deform model checkpoint路径进行恢复训练，None则自动查找latest
    # Deform model 参数
    temporal_encoding_dim: int = 16  # 时间编码维度
    lstm_input_dim: int = 19  # 3 (位置) + 16 (时间编码) = 19
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    graph_input_dim: int = 256
    graph_hidden_dim: int = 128
    graph_num_layers: int = 3
    graph_k: int = 16
    graph_dropout: float = 0.1
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 1  # 进一步减少batch size以适应全序列渲染
    num_dl_workers: int = 1  # 减少worker数量以节省内存
    gradient_accumulation_steps: int = 4  # 梯度累积步数，模拟更大的batch size
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    sequence_length: int = 4  # 进一步减少序列长度（最小有意义的序列）
    # 内存优化参数
    use_chunked_processing: bool = True  # 启用分块处理以节省内存
    chunk_size: int = 1000  # 使用更小的块大小以适应全序列渲染
    # 渲染参数
    render_full_sequence: bool = True  # 必须渲染整个序列以计算时序损失
    # 损失函数参数
    ssim_weight: float = 0.2  # SSIM损失的权重，L1损失权重为(1-ssim_weight)，总权重=1.0
    # 可视化参数
    port: int | None = None
    vis_debug: bool = False
    validate_every: int = 10
    save_every: int = 20
    save_videos_every: int = 50  # 保存视频的频率
    use_2dgs: bool = False
    devices: str = "0"  # CUDA设备索引，支持多GPU如"0,1"


class DeformTrainer:
    """Deform Model训练器"""
    
    def __init__(
        self,
        scene_model: SceneModel,
        deform_model: TemporalGraphRefinement,
        device: torch.device,
        config: DeformTrainConfig,
        train_dataset=None,  # 添加数据集引用以获取序列目标图像
    ):
        self.scene_model = scene_model
        self.deform_model = deform_model
        self.device = device
        self.config = config
        self.train_dataset = train_dataset
        
        # 冻结scene model
        for param in self.scene_model.parameters():
            param.requires_grad = False
        self.scene_model.eval()
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.deform_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        # 初始化梯度为零
        self.optimizer.zero_grad()
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.95
        )
        
        self.global_step = 0
        self.epoch = 0
        
        # 创建deformRefiner
        self.refiner = GaussianRefiner(self.deform_model, device)
        
        # 创建工作目录
        os.makedirs(config.work_dir, exist_ok=True)
        
        # 保存配置
        with open(f"{config.work_dir}/deform_config.yaml", "w") as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
    
    def temporal_encoding(self, timesteps, d_model=None):
        """
        使用正弦位置编码来表示时间信息
        
        Args:
            timesteps: 时间戳序列 (seq_len,) 
            d_model: 编码维度，默认使用config中的temporal_encoding_dim
            
        Returns:
            torch.Tensor: 时间编码 (seq_len, d_model)
        """
        if d_model is None:
            d_model = self.config.temporal_encoding_dim
            
        seq_len = len(timesteps)
        pe = torch.zeros(seq_len, d_model, device=self.device)
        
        # 归一化时间戳到[0, 1]范围
        max_time = self.train_dataset.num_frames - 1
        normalized_timesteps = timesteps.float() / max_time
        
        position = normalized_timesteps.unsqueeze(1)  # (seq_len, 1)
        
        # 计算不同频率的分量
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])  # 奇数维度情况
        else:
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
            
        return pe
    
    def extract_gaussian_sequences(self, batch) -> dict:
        """从scene model中提取高斯序列
        
        提取原始高斯参数和对应的时间戳序列，不进行预变换。
        变换将在渲染时进行。
        
        Args:
            batch: 包含以下键的字典:
                - imgs: 图像张量 (batch_size, H, W, C)
                - ts: 时间戳张量 (batch_size,)
        
        Returns:
            包含以下键的字典:
                - positions: 原始高斯位置 (batch_size, sequence_length, num_gaussians, 3)
                - orientations: 原始高斯方向 (batch_size, sequence_length, num_gaussians, 4) 
                - scales: 原始高斯尺度 (batch_size, sequence_length, num_gaussians, 3)
                - frame_ts: 序列帧时间戳 (batch_size, sequence_length)
                - target_images: 真实的序列目标图像 (batch_size, sequence_length, H, W, C)
        """
        batch_size = batch["imgs"].shape[0]
        sequence_length = self.config.sequence_length
        
        # 获取当前batch的时间戳
        ts = batch["ts"] # (batch_size,), shuffled timestamps
        
        # 获取原始高斯参数（不进行任何变换）
        original_means = self.scene_model.fg.params["means"]  # (num_gaussians, 3)
        original_quats = self.scene_model.fg.params["quats"]  # (num_gaussians, 4)
        original_scales = self.scene_model.fg.params["scales"]  # (num_gaussians, 3)
        num_gaussians = original_means.shape[0]
        
        sequences = {
            'positions': [], # (batch_size, sequence_length, num_gaussians, 3)
            'orientations': [], # (batch_size, sequence_length, num_gaussians, 4)
            'scales': [], # (batch_size, sequence_length, num_gaussians, 3)
            'frame_ts': [], # (batch_size, sequence_length) - 时间戳序列
            'target_images': [], # (batch_size, sequence_length, H, W, C)
        }
        
        for i in range(batch_size):
            center_t = ts[i].item()  # 这是数据集返回的时间戳
            
            # 安全检查：确保时间戳在有效范围内
            max_valid_t = self.train_dataset.num_frames - 1
            center_t = min(max(center_t, 0), max_valid_t)
            
            # 构建以当前帧为中心的时间序列
            if sequence_length > 1:
                half_seq = sequence_length // 2
                
                # 计算序列范围
                ideal_start = center_t - half_seq
                ideal_end = center_t + half_seq + (sequence_length % 2)
                
                # 处理边界情况
                if ideal_start < 0:
                    start_t = 0
                    end_t = min(sequence_length, self.train_dataset.num_frames)
                elif ideal_end >= self.train_dataset.num_frames:
                    end_t = self.train_dataset.num_frames
                    start_t = max(0, end_t - sequence_length)
                else:
                    start_t = ideal_start
                    end_t = ideal_end
                
                # 生成帧序列索引
                frame_indices = list(range(start_t, end_t))
            else:
                frame_indices = [center_t]
            
            # 为每个序列添加原始参数（所有帧使用相同的原始参数）
            frame_positions = []
            frame_orientations = []
            frame_scales = []
            frame_target_images = []
            frame_ts_list = []
            
            for frame_idx in frame_indices:
                # 确保frame_idx在有效范围内
                frame_idx = min(max(frame_idx, 0), max_valid_t)
                
                # 使用原始参数（不进行变换）
                frame_positions.append(original_means)  # (num_gaussians, 3)
                frame_orientations.append(original_quats)  # (num_gaussians, 4)
                frame_scales.append(original_scales)  # (num_gaussians, 3)
                frame_ts_list.append(frame_idx) # (sequence_length,)
                
                # 提取对应的目标图像
                target_image = self.train_dataset.get_image(frame_idx).to(self.device)  # (H, W, C)
                frame_target_images.append(target_image)
            
            # 堆叠为序列
            sequences['positions'].append(torch.stack(frame_positions, dim=0))
            sequences['orientations'].append(torch.stack(frame_orientations, dim=0))
            sequences['scales'].append(torch.stack(frame_scales, dim=0))
            sequences['frame_ts'].append(torch.tensor(frame_ts_list, device=self.device))
            sequences['target_images'].append(torch.stack(frame_target_images, dim=0))
        
        # 批量化序列
        for key in sequences:
            sequences[key] = torch.stack(sequences[key], dim=0)

        # print("sequences['frame_ts'].shape: ", sequences['frame_ts'].shape)
        # print("sequences['frame_ts']: ", sequences['frame_ts'])
        
        return sequences
    
    def render_with_deformed_gaussians(self, batch, frame_ts, deltas):
        """先变换高斯参数，再渲染高斯序列
        
        Args:
            batch: 包含图像、相机参数等的数据批次
                imgs: (batch_size, height, width, channels) 原始图像
                ts: (batch_size,) 每帧的时间戳
                w2cs: (batch_size, 4, 4) 世界坐标到相机坐标的变换矩阵
                Ks: (batch_size, 3, 3) 相机内参矩阵
            frame_ts: 序列帧时间戳 (batch_size, sequence_length)
            deltas: 变形增量字典
                delta_pos: (batch_size, seq_len, num_gaussians, 3) 位置增量
                delta_quat: (batch_size, seq_len, num_gaussians, 4) 方向增量
                delta_scale: (batch_size, seq_len, num_gaussians, 3) 尺度增量
                
        Returns:
            torch.Tensor: (batch_size, seq_len, H, W, C) 渲染的序列图像
        """
        batch_size = batch["imgs"].shape[0]
        # 从图像张量的形状中提取宽度和高度：batch["imgs"] 的形状为 (batch_size, height, width, channels)
        # 使用 shape[2:0:-1] 表示从索引 2 到索引 1（不包括0），并按步长 -1 反向切片
        # 即从 (B, H, W, C) 中提取出 (W, H)，用于表示图像的宽度和高度（注意顺序是 W, H）
        seq_len = frame_ts.shape[1]
        img_wh = batch["imgs"].shape[2:0:-1]

        # 查看batch中的Ks和w2cs内容
        # print("batch.Ks and w2cs:")
        # print(batch["Ks"])
        # print(batch["w2cs"])
        
        if self.config.render_full_sequence:
            # 渲染整个序列
            all_rendered_imgs = []
            
            for i in range(batch_size):
                batch_rendered_imgs = []
                w2c = batch["w2cs"][i]
                K = batch["Ks"][i]
                
                for seq_idx in range(seq_len):
                    current_t = frame_ts[i, seq_idx].item()
                    
                    # 1. 首先进行scene model的transformation
                    with torch.no_grad():
                        means, quats = self.scene_model.compute_poses_fg(
                            torch.tensor([current_t], device=self.device)
                        )
                        # means: (num_gaussians, 1, 3), quats: (num_gaussians, 1, 4)
                        transformed_means = means[:, 0, :]  # (num_gaussians, 3)
                        transformed_quats = quats[:, 0, :]  # (num_gaussians, 4)
                        transformed_scales = self.scene_model.fg.get_scales()  # (num_gaussians, 3)
                    
                    # 2. 然后应用变形增量
                    final_means = transformed_means + deltas['delta_pos'][i, seq_idx]
                    final_quats = self._apply_quat_delta(
                        transformed_quats, deltas['delta_quat'][i, seq_idx]
                    )
                    final_scales = transformed_scales * torch.exp(deltas['delta_scale'][i, seq_idx])
                    
                    # 保存原始参数
                    original_means = self.scene_model.fg.params["means"]
                    original_quats = self.scene_model.fg.params["quats"]
                    original_scales = self.scene_model.fg.params["scales"]
                    
                    # 3. 临时替换为最终参数进行渲染
                    self.scene_model.fg.params["means"] = final_means
                    self.scene_model.fg.params["quats"] = final_quats
                    self.scene_model.fg.params["scales"] = final_scales
                    
                    try:
                        # 渲染时不再调用compute_poses_fg，直接使用设置的参数
                        # 传入None作为时间戳，避免重复变换
                        rendered = self.scene_model.render(
                            None,  # 传入None避免重复变换
                            w2c[None],
                            K[None],
                            img_wh,
                            return_depth=False,
                            return_mask=False,
                            fg_only=False # 渲染完整场景，包含背景
                        )
                        
                        # 立即提取图像并释放rendered字典以节省内存
                        img = rendered["img"][0].clone() # 去掉batch维度得到 (H, W, 3)
                        del rendered
                        batch_rendered_imgs.append(img)
                        
                    finally:
                        # 恢复原始参数
                        self.scene_model.fg.params["means"] = original_means
                        self.scene_model.fg.params["quats"] = original_quats
                        self.scene_model.fg.params["scales"] = original_scales
                
                all_rendered_imgs.append(torch.stack(batch_rendered_imgs, dim=0))
            
            return torch.stack(all_rendered_imgs, dim=0)  # (batch_size, seq_len, H, W, C)
         
    def train_step(self, batch):
        """单步训练
        batch:
            imgs: (batch_size, height, width, channels) 原始图像
            ts: (batch_size,) 每帧的时间戳
            w2cs: (batch_size, 4, 4) 世界坐标到相机坐标的变换矩阵
            Ks: (batch_size, 3, 3) 相机内参矩阵
            ...
        """
        import time
        step_start = time.time()
        
        self.deform_model.train()
        
        # 清理GPU缓存以防止内存累积
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        
        # 1. 提取高斯序列
        seq_start = time.time()
        # 返回的gaussian_sequences是一个字典，包含四个键：positions, orientations, scales, target_images
        # 每个键的值是一个张量，形状为 (batch_size, sequence_length, num_gaussians, dim)
        gaussian_sequences = self.extract_gaussian_sequences(batch)
        seq_time = time.time() - seq_start
        # print("seq_extract_time: ", seq_time)
        
        # 2. 使用deform model优化序列
        deform_start = time.time()
        # 需要重新组织数据格式以适应deform model
        batch_size, seq_len, num_gaussians, _ = gaussian_sequences['positions'].shape
        
        # 内存监控：打印当前GPU内存使用情况
        if self.global_step % 50 == 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            guru.info(f"Step {self.global_step}: GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        # 转换为deform model期望的格式：交换sequence_length和num_gaussians维度
        positions_input = gaussian_sequences['positions'].transpose(1, 2)  # (batch_size, num_gaussians, seq_len, 3)
        
        # 获取deltas
        deltas_sequence = {
            'delta_pos': [],
            'delta_quat': [],
            'delta_scale': [],
        }
        
        
        for b in range(batch_size):
            sample_positions = positions_input[b]  # (num_gaussians, seq_len, 3) - 单个样本的所有高斯点位置序列
            
            # 获取时间编码
            frame_timesteps = gaussian_sequences['frame_ts'][b]  # (seq_len,)
            time_encoding = self.temporal_encoding(frame_timesteps)  # (seq_len, temporal_encoding_dim)
            
            if self.config.use_chunked_processing:
                # 分块处理高斯点（节省内存）
                chunk_size = self.config.chunk_size
                sample_delta_pos = []
                sample_delta_quat = []
                sample_delta_scale = []
                
                for chunk_start in range(0, num_gaussians, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_gaussians)
                    chunk_positions = sample_positions[chunk_start:chunk_end]  # (chunk_size, seq_len, 3)
                    
                    # 为每个高斯点复制时间编码
                    chunk_size_actual = chunk_positions.shape[0] # chunked高斯点数量
                    chunk_time_encoding = time_encoding.unsqueeze(0).expand(chunk_size_actual, -1, -1)  
                    # (chunk_size, seq_len, temporal_encoding_dim)
                    
                    # 拼接位置和时间编码
                    chunk_input = torch.cat([chunk_positions, chunk_time_encoding], dim=-1)  
                    # (chunk_size, seq_len, 3 + temporal_encoding_dim)
                    
                    # 前向传播 - 直接获取deltas
                    deltas = self.deform_model(chunk_input)
                    # return: delta_pos, delta_quat, delta_scale
                    # shape: (chunk_size, seq_len, 3), (chunk_size, seq_len, 4), (chunk_size, seq_len, 3)
                    
                    sample_delta_pos.append(deltas['delta_pos'])
                    sample_delta_quat.append(deltas['delta_quat'])
                    sample_delta_scale.append(deltas['delta_scale'])
                
                # 合并块并调整维度顺序
                deltas_sequence['delta_pos'].append(torch.cat(sample_delta_pos, dim=0).transpose(0, 1)) # 交换num_gaussians和seq_len维度
                # shape: (batch_size, seq_len, num_gaussians, 3)
                deltas_sequence['delta_quat'].append(torch.cat(sample_delta_quat, dim=0).transpose(0, 1)) # 交换num_gaussians和seq_len维度
                # shape: (batch_size, seq_len, num_gaussians, 4)    
                deltas_sequence['delta_scale'].append(torch.cat(sample_delta_scale, dim=0).transpose(0, 1)) # 交换num_gaussians和seq_len维度
                # shape: (batch_size, seq_len, num_gaussians, 3)
                
            else:
                # 不分块处理
                # 为所有高斯点复制时间编码
                sample_time_encoding = time_encoding.unsqueeze(0).expand(num_gaussians, -1, -1)  # (num_gaussians, seq_len, temporal_encoding_dim)
                
                # 拼接位置和时间编码
                sample_input = torch.cat([sample_positions, sample_time_encoding], dim=-1)  # (num_gaussians, seq_len, 3 + temporal_encoding_dim)
                
                deltas = self.deform_model(sample_input)
                
                # 直接添加结果，调整维度顺序
                deltas_sequence['delta_pos'].append(deltas['delta_pos'].transpose(0, 1))
                deltas_sequence['delta_quat'].append(deltas['delta_quat'].transpose(0, 1))
                deltas_sequence['delta_scale'].append(deltas['delta_scale'].transpose(0, 1))
                   
        # 堆叠批次
        for key in deltas_sequence:
            deltas_sequence[key] = torch.stack(deltas_sequence[key], dim=0)
            # key: delta_pos, delta_quat, delta_scale
            # shape: (batch_size, seq_len, num_gaussians, dim)
        
        deform_time = time.time() - deform_start
        # print("deform_time: ", deform_time)
        
        # 3. 渲染并计算损失
        render_start = time.time()
        # 直接使用deltas_sequence进行渲染
        with torch.cuda.amp.autocast(enabled=False):  # 禁用自动混合精度以避免潜在的内存问题
            rendered_imgs = self.render_with_deformed_gaussians(batch, gaussian_sequences['frame_ts'], deltas_sequence)
        render_time = time.time() - render_start
        # print("render_time: ", render_time)

        # 4. 损失计算
        loss_start = time.time()
        if self.config.render_full_sequence:
            # 渲染了整个序列: rendered_imgs.shape = (B, T, H, W, 3)
            # 使用真实的序列目标图像
            target_imgs = gaussian_sequences['target_images']  # (B, T, H, W, 3)
            
            # 验证我们确实在使用真实的序列目标（而不是重复的中心帧）
            if hasattr(self, 'global_step') and self.global_step % 50 == 0:
                # 计算序列中不同帧之间的差异，确保它们不是相同的
                seq_len = target_imgs.shape[1]
                if seq_len > 1:
                    frame_diffs = []
                    for t in range(1, seq_len):
                        diff = torch.mean(torch.abs(target_imgs[:, t] - target_imgs[:, 0]))
                        frame_diffs.append(diff.item())
                    avg_frame_diff = sum(frame_diffs) / len(frame_diffs)
                    guru.info(f"Step {self.global_step}: Average frame difference in target sequence: {avg_frame_diff:.6f}")
                    if avg_frame_diff < 1e-6:
                        guru.warning("Target images appear to be identical across sequence! Check if dataset \
                                     is providing correct frames.")
            
            # 确保形状匹配
            assert rendered_imgs.shape == target_imgs.shape, \
                f"Shape mismatch: rendered_imgs {rendered_imgs.shape} vs target_imgs {target_imgs.shape}"
            
            # RGB损失 - 使用L1损失和SSIM损失的组合
            l1_rgb_loss = l1_loss(rendered_imgs, target_imgs)
            
            # SSIM需要(N, C, H, W)格式，需要重塑张量
            # rendered_imgs和target_imgs的形状都是(B, T, H, W, C)
            B, T, H, W, C = rendered_imgs.shape
            rendered_ssim = rendered_imgs.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)  # (B*T, C, H, W)
            target_ssim = target_imgs.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)  # (B*T, C, H, W)
            
            ssim_rgb_loss = 1.0 - ssim(rendered_ssim, target_ssim)
            rgb_loss = (1 - self.config.ssim_weight) * l1_rgb_loss + self.config.ssim_weight * ssim_rgb_loss    
            
        loss_time = time.time() - loss_start
        # print("loss_time: ", loss_time)

        # 5. 反向传播
        backward_start = time.time()
        # 将损失除以累积步数以获得平均梯度
        rgb_loss = rgb_loss / self.config.gradient_accumulation_steps
        rgb_loss.backward()
        
        # 每隔gradient_accumulation_steps步或在最后一步更新参数
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.deform_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        backward_time = time.time() - backward_start
        # print("backward_time: ", backward_time)
        total_time = time.time() - step_start
        # print("total_time: ", total_time)

        # 打印各步骤耗时
        # if self.global_step % 10 == 0:
        #     guru.info(f"⏱️  Step {self.global_step} Timing:")
        #     guru.info(f"  1. Extract sequences: {seq_time:.3f}s ({seq_time/total_time*100:.1f}%)")
        #     guru.info(f"  2. Deform model: {deform_time:.3f}s ({deform_time/total_time*100:.1f}%)")
        #     guru.info(f"  3. Render: {render_time:.3f}s ({render_time/total_time*100:.1f}%)")
        #     guru.info(f"  4. Loss calc: {loss_time:.3f}s ({loss_time/total_time*100:.1f}%)")
        #     guru.info(f"  5. Backward: {backward_time:.3f}s ({backward_time/total_time*100:.1f}%)")
        #     guru.info(f"  Total: {total_time:.3f}s")
        
        self.global_step += 1
        
        # 返回损失组件信息
        loss_info = {
            'total_loss': rgb_loss.item(),
            'l1_loss': l1_rgb_loss.item(),
            'ssim_loss': ssim_rgb_loss.item(),
        }
        
        return loss_info
    
    def _apply_quat_delta(self, original_quat, delta_quat):
        """应用四元数增量"""
        updated_quat = original_quat + delta_quat
        return torch.nn.functional.normalize(updated_quat, p=2, dim=-1)
    
    def create_deformed_scene_model(self):
        """创建一个集成了变形模型的SceneModel包装器"""
        class DeformedSceneModel:
            def __init__(self, scene_model, deform_model, trainer):
                self.scene_model = scene_model # 来自trainer.scene_model
                self.deform_model = deform_model # 来自trainer.deform_model
                self.trainer = trainer # 来自trainer
                # 复制scene_model的属性以保持兼容性
                self.has_bg = scene_model.has_bg
                self.num_gaussians = scene_model.num_gaussians
                self.num_fg_gaussians = scene_model.num_fg_gaussians
                self.num_bg_gaussians = scene_model.num_bg_gaussians
                 
            def eval(self):
                """设置为评估模式"""
                self.deform_model.eval()
                return self
                 
            def train(self, mode=True):
                """设置训练模式"""
                self.deform_model.train(mode)
                return self
                 
            def render(self, t, w2c, K, img_wh, **kwargs):
                """渲染单帧图像
                
                逻辑顺序：
                0. 获取当前帧附近的序列索引
                1. 先对原始高斯参数进行时间变换（transformation）
                2. 然后基于变换后的参数计算变形增量（deform）
                3. 最后应用变形增量并渲染
                """
                # 只对单个时间戳进行变形
                with torch.no_grad():
                    # 0. 构建以当前帧为中心的序列索引
                    sequence_length = self.trainer.config.sequence_length
                    if sequence_length > 1:
                        half_seq = sequence_length // 2
                        max_valid_t = self.trainer.train_dataset.num_frames - 1
                        t = min(max(t, 0), max_valid_t)  # 确保t在有效范围内
                        
                        # 计算序列范围
                        ideal_start = t - half_seq
                        ideal_end = t + half_seq + (sequence_length % 2)
                        
                        # 处理边界情况
                        if ideal_start < 0:
                            start_t = 0
                            end_t = min(sequence_length, self.trainer.train_dataset.num_frames)
                        elif ideal_end >= self.trainer.train_dataset.num_frames:
                            end_t = self.trainer.train_dataset.num_frames
                            start_t = max(0, end_t - sequence_length)
                        else:
                            start_t = ideal_start
                            end_t = ideal_end
                        
                        # 生成帧序列索引
                        frame_indices = list(range(start_t, end_t))
                        # 找到当前帧在序列中的位置
                        current_frame_idx = frame_indices.index(t)
                    else:
                        frame_indices = [t]
                        current_frame_idx = 0
                    
                    # 1. 获取序列中所有帧的transformed参数
                    all_transformed_means = [] # (seq_len, num_gaussians, 3)    
                    all_transformed_quats = [] # (seq_len, num_gaussians, 4)
                    all_transformed_scales = [] # (seq_len, num_gaussians, 3)
                    
                    for frame_t in frame_indices:
                        means, quats = self.scene_model.compute_poses_fg(
                            torch.tensor([frame_t], device=self.trainer.device)
                        )
                        all_transformed_means.append(means[:, 0, :]) 
                        all_transformed_quats.append(quats[:, 0, :])  
                        all_transformed_scales.append(self.scene_model.fg.get_scales())  
                    
                    # 堆叠为序列
                    transformed_means_seq = torch.stack(all_transformed_means, dim=0)  # (seq_len, num_gaussians, 3)
                    transformed_quats_seq = torch.stack(all_transformed_quats, dim=0)  # (seq_len, num_gaussians, 4)
                    transformed_scales_seq = torch.stack(all_transformed_scales, dim=0)  # (seq_len, num_gaussians, 3)
                    num_gaussians = transformed_means_seq.shape[1]
                    
                    # 2. 构造变形模型的输入（使用transformed的位置序列）
                    positions = transformed_means_seq.unsqueeze(0)  # (1, seq_len, num_gaussians, 3)
                    frame_ts = torch.tensor(frame_indices, device=self.trainer.device)  # (seq_len,)
                    
                    # 3. 获取时间编码
                    time_encoding = self.trainer.temporal_encoding(frame_ts)  # (seq_len, temporal_encoding_dim)
                    
                    # 4. 应用变形模型
                    positions_input = positions.transpose(1, 2)  # (1, num_gaussians, seq_len, 3)
                    sample_positions = positions_input[0]  # (num_gaussians, seq_len, 3)
                    
                    # 使用分块处理以节省内存（与训练时一致）
                    if self.trainer.config.use_chunked_processing:
                        chunk_size = self.trainer.config.chunk_size
                        delta_pos_chunks = []
                        delta_quat_chunks = []
                        delta_scale_chunks = []
                        
                        for chunk_start in range(0, num_gaussians, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, num_gaussians)
                            chunk_positions = sample_positions[chunk_start:chunk_end]  # (chunk_size, seq_len, 3)
                            
                            # 为每个高斯点复制时间编码
                            chunk_size_actual = chunk_positions.shape[0]
                            chunk_time_encoding = time_encoding.unsqueeze(0).expand(chunk_size_actual, -1, -1)  # (chunk_size, seq_len, temporal_encoding_dim)
                            
                            # 拼接位置和时间编码
                            chunk_input = torch.cat([chunk_positions, chunk_time_encoding], dim=-1)  # (chunk_size, seq_len, 3 + temporal_encoding_dim)
                            
                            # 前向传播获取变形增量
                            chunk_deltas = self.deform_model(chunk_input)
                            # 只取当前帧的增量
                            delta_pos_chunks.append(chunk_deltas['delta_pos'][:, current_frame_idx, :])  # (chunk_size, 3)
                            delta_quat_chunks.append(chunk_deltas['delta_quat'][:, current_frame_idx, :])  # (chunk_size, 4)
                            delta_scale_chunks.append(chunk_deltas['delta_scale'][:, current_frame_idx, :])  # (chunk_size, 3)
                        
                        delta_pos = torch.cat(delta_pos_chunks, dim=0)  # (num_gaussians, 3)
                        delta_quat = torch.cat(delta_quat_chunks, dim=0)  # (num_gaussians, 4)
                        delta_scale = torch.cat(delta_scale_chunks, dim=0)  # (num_gaussians, 3)
                    else:
                        # 为每个高斯点复制时间编码
                        sample_time_encoding = time_encoding.unsqueeze(0).expand(num_gaussians, -1, -1)  # (num_gaussians, seq_len, temporal_encoding_dim)
                        
                        # 拼接位置和时间编码
                        sample_input = torch.cat([sample_positions, sample_time_encoding], dim=-1)  # (num_gaussians, seq_len, 3 + temporal_encoding_dim)
                        
                        # 前向传播获取变形增量
                        deltas = self.deform_model(sample_input)
                        # 只取当前帧的增量
                        delta_pos = deltas['delta_pos'][:, current_frame_idx, :]  # (num_gaussians, 3)
                        delta_quat = deltas['delta_quat'][:, current_frame_idx, :]  # (num_gaussians, 4)
                        delta_scale = deltas['delta_scale'][:, current_frame_idx, :]  # (num_gaussians, 3)
                    
                    # 5. 应用变形增量到当前帧的transformed参数上
                    transformed_means = transformed_means_seq[current_frame_idx]  # (num_gaussians, 3)
                    transformed_quats = transformed_quats_seq[current_frame_idx]  # (num_gaussians, 4)
                    transformed_scales = transformed_scales_seq[current_frame_idx]  # (num_gaussians, 3)
                    
                    final_means = transformed_means + delta_pos
                    final_quats = self.trainer._apply_quat_delta(transformed_quats, delta_quat)
                    final_scales = transformed_scales * torch.exp(delta_scale)
                    
                    # 6. 临时替换参数并渲染
                    original_means_backup = self.scene_model.fg.params["means"]
                    original_quats_backup = self.scene_model.fg.params["quats"]
                    original_scales_backup = self.scene_model.fg.params["scales"]
                    
                    try:
                        self.scene_model.fg.params["means"] = final_means
                        self.scene_model.fg.params["quats"] = final_quats
                        self.scene_model.fg.params["scales"] = final_scales
                        
                        # 渲染时传入None避免重复变换
                        return self.scene_model.render(None, w2c, K, img_wh, **kwargs)
                    finally:
                        # 恢复原始参数
                        self.scene_model.fg.params["means"] = original_means_backup
                        self.scene_model.fg.params["quats"] = original_quats_backup
                        self.scene_model.fg.params["scales"] = original_scales_backup
        
        return DeformedSceneModel(self.scene_model, self.deform_model, self)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'deform_model_state_dict': self.deform_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': asdict(self.config)
        }
        torch.save(checkpoint, path)
        guru.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.deform_model.load_state_dict(checkpoint['deform_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        guru.info(f"Checkpoint loaded from {path}")


def main(cfg: DeformTrainConfig):
    # 创建工作目录
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.devices
        device = torch.device("cuda:0")  # 使用第一个可见设备
        guru.info(f"Using CUDA devices: {cfg.devices}, mapped to cuda:0")
    else:
        device = torch.device("cpu")
        guru.info("CUDA not available, using CPU")
    
    # 加载数据
    train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
        get_train_val_datasets(cfg.data, load_val=True)
    )
    guru.info(f"Training dataset has {train_dataset.num_frames} frames")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_dl_workers,
        shuffle=True,  # 可以shuffle，因为序列是独立构建的
        collate_fn=BaseDataset.train_collate_fn,
    )
    # train_loader 中的dataset 包含 frame_names, time_ids, imgs, masks, tracks, Ks, w2cs等
    
    # 加载预训练的scene model
    guru.info(f"Loading scene model from {cfg.scene_model_path}")
    scene_ckpt = torch.load(cfg.scene_model_path, map_location=device)
    scene_model = SceneModel.init_from_state_dict(scene_ckpt["model"])
    scene_model = scene_model.to(device)
    scene_model.use_2dgs = cfg.use_2dgs
    
    # 创建deform model
    deform_model = TemporalGraphRefinement(
        lstm_input_dim=cfg.lstm_input_dim,
        lstm_hidden_dim=cfg.lstm_hidden_dim,
        lstm_num_layers=cfg.lstm_num_layers,
        lstm_dropout=cfg.lstm_dropout,
        graph_input_dim=cfg.graph_input_dim,
        graph_hidden_dim=cfg.graph_hidden_dim,
        graph_num_layers=cfg.graph_num_layers,
        graph_k=cfg.graph_k,
        graph_dropout=cfg.graph_dropout,
    ).to(device)
    
    # 创建训练器
    trainer = DeformTrainer(scene_model, deform_model, device, cfg, train_dataset)
    
    # 检查并加载checkpoint
    start_epoch = 0
    if cfg.resume_from is not None:
        # 用户指定了特定的checkpoint路径
        if os.path.exists(cfg.resume_from):
            guru.info(f"Loading specified checkpoint: {cfg.resume_from}")
            trainer.load_checkpoint(cfg.resume_from)
            start_epoch = trainer.epoch + 1
            guru.info(f"Resumed training from epoch {start_epoch}, step {trainer.global_step}")
        else:
            guru.error(f"Specified checkpoint not found: {cfg.resume_from}")
            raise FileNotFoundError(f"Checkpoint not found: {cfg.resume_from}")
    else:
        # 自动查找最新的checkpoint
        checkpoint_path = f"{cfg.work_dir}/deform_checkpoint_latest.pth"
        if os.path.exists(checkpoint_path):
            guru.info(f"Found latest checkpoint at {checkpoint_path}, loading...")
            trainer.load_checkpoint(checkpoint_path)
            start_epoch = trainer.epoch + 1
            guru.info(f"Resumed training from epoch {start_epoch}, step {trainer.global_step}")
        else:
            guru.info("No checkpoint found, starting training from scratch")
    
    # 创建验证器（使用flow3d.validator.Validator）
    validator = None
    if (
        train_video_view is not None
        or val_img_dataset is not None
        or val_kpt_dataset is not None
    ):
        # 创建集成了变形模型的SceneModel包装器
        deformed_scene_model = trainer.create_deformed_scene_model()
        
        validator = Validator(
            model=deformed_scene_model,
            device=device,
            train_loader=(
                DataLoader(train_video_view, batch_size=1) if train_video_view else None
            ),
            val_img_loader=(
                DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
            ),
            val_kpt_loader=(
                DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
            ),
            save_dir=cfg.work_dir,
        )
    
    # 训练循环
    guru.info(f"Starting deform model training from epoch {start_epoch}...")
    for epoch in tqdm(range(start_epoch, cfg.num_epochs), desc="Epochs", initial=start_epoch, total=cfg.num_epochs):
        trainer.epoch = epoch
        
        # 训练
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = to_device(batch, device)
            loss_info = trainer.train_step(batch)
            total_loss += loss_info['total_loss']
            num_batches += 1
            
            if trainer.global_step % 10 == 0:
                guru.info(f"Step {trainer.global_step}, Total Loss: {loss_info['total_loss']:.6f}, "
                         f"L1: {loss_info['l1_loss']:.6f}, SSIM: {loss_info['ssim_loss']:.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        guru.info(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")
        
        # 验证
        # 每validate_every个epoch进行一次验证，并且需要validator存在
        if validator is not None and (epoch + 1) % cfg.validate_every == 0:
            # 设置变形模型为评估模式
            trainer.deform_model.eval()
            val_logs = validator.validate()
            guru.info(f"Epoch {epoch} - Validation metrics: {val_logs}")
            # 恢复训练模式
            trainer.deform_model.train()
            
        # 保存视频（可选）
        if validator is not None and (epoch > 0 and (epoch + 1) % cfg.save_videos_every == 0) or (epoch == cfg.num_epochs - 1):
            trainer.deform_model.eval()
            validator.save_train_videos(epoch)
            trainer.deform_model.train()
        
        # 保存检查点
        if (epoch + 1) % cfg.save_every == 0:
            checkpoint_path = f"{cfg.work_dir}/deform_checkpoint_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(checkpoint_path)
        
        # 始终保存最新的checkpoint（用于断点续训）
        latest_checkpoint_path = f"{cfg.work_dir}/deform_checkpoint_latest.pth"
        trainer.save_checkpoint(latest_checkpoint_path)
        
        # 更新学习率
        trainer.scheduler.step()
    
    # 保存最终模型
    final_path = f"{cfg.work_dir}/deform_model_final.pth"
    trainer.save_checkpoint(final_path)
    guru.success(f"Training completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main(tyro.cli(DeformTrainConfig)) 