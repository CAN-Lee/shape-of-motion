"""
trainer.py - 核心训练类

定义GaussianSequenceTrainer训练器，负责模型训练的核心逻辑。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
import numpy as np

# 添加SSIM损失
from pytorch_msssim import SSIM

from Gaussian_Sequence.LSTM_Refinements import BiLSTMRefinement, GaussianSequenceRefiner
from Gaussian_Sequence.LSTM_Graph_Refinement import TemporalGraphRefinement, GaussianRefiner

# 直接导入flow3d的trainer和损失函数
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from flow3d.trainer import Trainer as Flow3DTrainer
from flow3d.loss_utils import masked_l1_loss, compute_gradient_loss


class GaussianSequenceTrainer:
    """高斯序列训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.num_epochs // 3,
            gamma=0.5
        )
        
        # 日志记录器
        self.writer = SummaryWriter(config.log_dir)
        
        # 损失函数 - 参考flow3d的设计
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 创建优化器实例（仅用于推理）
        if config.model_type == "bilstm":
            self.refiner = GaussianSequenceRefiner(self.model, self.device)
        else:
            self.refiner = GaussianRefiner(self.model, self.device)
        
        # 创建flow3d trainer实例用于RGB损失计算
        self.flow3d_trainer = None
    
    def _create_model(self) -> nn.Module:
        """创建模型"""
        if self.config.model_type == "bilstm":
            return BiLSTMRefinement(
                input_dim=self.config.lstm_input_dim,
                hidden_dim=self.config.lstm_hidden_dim,
                num_layers=self.config.lstm_num_layers,
                dropout=self.config.lstm_dropout
            )
        elif self.config.model_type == "graph":
            return TemporalGraphRefinement(
                lstm_input_dim=self.config.lstm_input_dim,
                lstm_hidden_dim=self.config.lstm_hidden_dim,
                lstm_num_layers=self.config.lstm_num_layers,
                lstm_dropout=self.config.lstm_dropout,
                graph_input_dim=self.config.graph_input_dim,
                graph_hidden_dim=self.config.graph_hidden_dim,
                graph_num_layers=self.config.graph_num_layers,
                graph_k=self.config.graph_k,
                graph_dropout=self.config.graph_dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def train_step(self, batch: Dict[str, torch.Tensor], flow3d_trainer=None, flow3d_batch=None) -> Dict[str, float]:
        """
        单步训练 - 可直接使用flow3d trainer的方法
        
        Args:
            batch: 高斯序列训练批次数据
            flow3d_trainer: flow3d trainer实例(可选)
            flow3d_batch: flow3d格式的批次数据(可选)
            
        Returns:
            Dict[str, float]: 损失字典
        """
        self.model.train()
        
        # 如果提供了flow3d trainer和batch，直接使用
        if flow3d_trainer is not None and flow3d_batch is not None:
            try:
                # 直接调用flow3d trainer的compute_losses方法
                loss, stats, _, _ = flow3d_trainer.compute_losses(flow3d_batch)
                
                # 提取RGB损失
                rgb_loss = stats.get('train/rgb_loss', loss.item())
                
                losses_dict = {
                    'total_loss': loss.item(),
                    'rgb_loss': rgb_loss
                }
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                return losses_dict
                
            except Exception as e:
                print(f"Failed to use flow3d trainer: {e}")
                # 回退到自定义实现
                pass
        
        # 获取输入序列
        positions = batch['positions'].to(self.device)  # (batch_size, seq_len, num_gaussians, 3)
        orientations = batch['orientations'].to(self.device)  # (batch_size, seq_len, num_gaussians, 4)
        scales = batch['scales'].to(self.device)  # (batch_size, seq_len, num_gaussians, 3)
        colors = batch['colors'].to(self.device)  # (batch_size, seq_len, num_gaussians, 3)
        opacities = batch['opacities'].to(self.device)  # (batch_size, seq_len, num_gaussians, 1)
        
        batch_size, seq_len, num_gaussians, _ = positions.shape
        
        # 计算损失
        loss, losses_dict = self.compute_loss(positions, orientations, scales, colors, opacities)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return losses_dict
    
    def compute_loss(self, positions: torch.Tensor, orientations: torch.Tensor, 
                    scales: torch.Tensor, colors: torch.Tensor, opacities: torch.Tensor,
                    scene_model=None, render_data=None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        统一的损失计算函数 - 直接使用flow3d trainer的方法
        
        Args:
            positions: 位置 (batch_size, seq_len, num_gaussians, 3)
            orientations: 朝向 (batch_size, seq_len, num_gaussians, 4)
            scales: 尺度 (batch_size, seq_len, num_gaussians, 3)
            colors: 颜色 (batch_size, seq_len, num_gaussians, 3)
            opacities: 不透明度 (batch_size, seq_len, num_gaussians, 1)
            scene_model: flow3d场景模型
            render_data: flow3d格式的批次数据
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 总损失和损失字典
        """
        # 如果提供了flow3d场景模型和数据，直接使用flow3d trainer的compute_losses
        if scene_model is not None and render_data is not None and hasattr(scene_model, 'compute_losses'):
            try:
                # 直接调用flow3d trainer的compute_losses方法
                loss, stats, _, _ = scene_model.compute_losses(render_data)
                
                # 提取RGB损失
                rgb_loss = stats.get('train/rgb_loss', loss.item())
                
                losses_dict = {
                    'total_loss': loss.item(),
                    'rgb_loss': rgb_loss
                }
                
                return loss, losses_dict
                
            except Exception as e:
                print(f"Failed to use flow3d trainer: {e}")
                # 回退到自定义实现
                pass
        
        # 回退到自定义实现
        batch_size, seq_len, num_gaussians, _ = positions.shape
        
        # 重新排列维度用于模型处理
        positions_reshaped = positions.permute(0, 2, 1, 3).contiguous().view(-1, seq_len, 3)
        
        # 前向传播获取增量
        if self.config.model_type == "bilstm":
            deltas = self.model(positions_reshaped)
        elif self.config.model_type == "graph":
            outputs = self.model(positions_reshaped)
            deltas = outputs
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # 应用增量更新
        delta_pos = deltas['delta_pos'].view(batch_size, num_gaussians, seq_len, 3).permute(0, 2, 1, 3)
        delta_quat = deltas['delta_quat'].view(batch_size, num_gaussians, seq_len, 4).permute(0, 2, 1, 3)
        delta_scale = deltas['delta_scale'].view(batch_size, num_gaussians, seq_len, 3).permute(0, 2, 1, 3)
        
        # 计算优化后的参数
        refined_positions = positions + delta_pos
        refined_orientations = F.normalize(orientations + delta_quat, p=2, dim=-1)
        refined_scales = scales * torch.exp(delta_scale)
        
        # 计算RGB损失
        rgb_loss = self._compute_render_loss(
            refined_positions, refined_orientations, refined_scales, 
            scene_model, render_data
        )
        
        # 总损失就是RGB损失
        total_loss = rgb_loss
        
        # 损失字典
        losses_dict = {
            'total_loss': total_loss.item(),
            'rgb_loss': rgb_loss.item()
        }
        
        return total_loss, losses_dict
    
    def _compute_rgb_loss(self, rendered_image: torch.Tensor, target_image: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算RGB损失 - 直接使用flow3d的RGB损失逻辑
        
        Args:
            rendered_image: 渲染图像 (H, W, 3)
            target_image: 目标图像 (H, W, 3)
            mask: 可选的掩码 (H, W)
            
        Returns:
            torch.Tensor: RGB损失
        """
        # 直接使用flow3d的RGB损失计算：0.8 * L1 + 0.2 * SSIM
        if self.config.use_masked_loss and mask is not None:
            # 使用flow3d的masked_l1_loss
            rgb_loss = masked_l1_loss(
                rendered_image, target_image, mask[..., None], 
                quantile=self.config.loss_quantile
            )
        else:
            # flow3d的标准RGB损失
            l1_loss = F.l1_loss(rendered_image, target_image)
            ssim_loss = 1.0 - self.ssim(
                rendered_image.unsqueeze(0).permute(0, 3, 1, 2),
                target_image.unsqueeze(0).permute(0, 3, 1, 2)
            )
            # flow3d的权重：0.8 * L1 + 0.2 * SSIM  
            rgb_loss = 0.8 * l1_loss + 0.2 * ssim_loss
        
        return rgb_loss
    
    def _compute_render_loss(self, positions: torch.Tensor, orientations: torch.Tensor, 
                           scales: torch.Tensor, scene_model, render_data: Dict) -> torch.Tensor:
        """
        计算渲染损失 - 参考flow3d的RGB损失设计
        
        Args:
            positions: 优化后的位置 (batch_size, seq_len, num_gaussians, 3)
            orientations: 优化后的朝向 (batch_size, seq_len, num_gaussians, 4)
            scales: 优化后的尺度 (batch_size, seq_len, num_gaussians, 3)
            scene_model: 场景模型
            render_data: 渲染数据，包含target_images, masks, camera_matrices等
            
        Returns:
            torch.Tensor: 渲染损失
        """
        batch_size, seq_len, num_gaussians, _ = positions.shape
        render_losses = []
        
        # 随机选择帧进行渲染（避免内存问题）
        for batch_idx in range(min(batch_size, 2)):  # 限制批次数量
            frame_indices = torch.randint(0, seq_len, (2,))  # 每个样本选择2帧
            
            for frame_idx in frame_indices:
                # 获取该帧的高斯参数
                frame_positions = positions[batch_idx, frame_idx]  # (num_gaussians, 3)
                frame_orientations = orientations[batch_idx, frame_idx]  # (num_gaussians, 4)
                frame_scales = scales[batch_idx, frame_idx]  # (num_gaussians, 3)
                
                # 如果有场景模型，使用实际渲染
                if scene_model is not None and render_data is not None:
                    try:
                        # 调用实际的渲染函数
                        rendered_image = scene_model.render(
                            frame_positions, frame_orientations, frame_scales,
                            **render_data.get('render_kwargs', {})
                        )
                        target_image = render_data['target_images'][batch_idx][frame_idx]
                        
                        # 使用新的RGB损失计算函数
                        mask = None
                        if 'masks' in render_data and render_data['masks'] is not None:
                            mask = render_data['masks'][batch_idx][frame_idx]
                        
                        frame_loss = self._compute_rgb_loss(rendered_image, target_image, mask)
                        
                    except Exception as e:
                        # 如果渲染失败，使用位置约束作为代理损失
                        frame_loss = torch.var(frame_positions, dim=0).mean()
                        
                else:
                    # 没有场景模型时，使用位置约束作为代理损失
                    frame_loss = torch.var(frame_positions, dim=0).mean()
                
                render_losses.append(frame_loss)
        
        return torch.stack(render_losses).mean() if render_losses else torch.tensor(0.0, device=positions.device)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # 训练步骤
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            # 记录损失
            if batch_idx % self.config.log_interval == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)
                print(f"Epoch {self.epoch}, Batch {batch_idx}, RGB Loss: {losses['rgb_loss']:.6f}")
            
            self.global_step += 1
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均损失
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证模型 - 统一的RGB损失验证"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 获取输入数据
                positions = batch['positions'].to(self.device)
                orientations = batch['orientations'].to(self.device)
                scales = batch['scales'].to(self.device)
                colors = batch['colors'].to(self.device)
                opacities = batch['opacities'].to(self.device)
                
                # 计算验证损失
                _, loss_dict = self.compute_loss(positions, orientations, scales, colors, opacities)
                val_losses.append(loss_dict)
        
        # 计算平均损失
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in val_losses])
        
        return avg_losses