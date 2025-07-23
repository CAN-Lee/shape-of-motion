"""
基于 Bi-LSTM 的 Refinement of Dynamic Gaussian Sequences (Refinements.py)

输入所有前景Initial Dynamic Gaussian Sequences的Gaussian position(xyz)，
基于Bi-LSTM来进一步优化，输出每一帧前景高斯的增量更新(delta_xyz, delta_orientation, delta_scale)。

此模块专注于推理时的高斯序列优化，训练功能在 trainer.py 中实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class BiLSTMRefinement(nn.Module):
    """基于双向LSTM的高斯序列优化网络"""
    
    def __init__(
        self,
        input_dim: int = 3,  # 输入位置的维度 (x, y, z)
        hidden_dim: int = 256,  # LSTM隐藏状态维度
        num_layers: int = 2,  # LSTM层数
        dropout: float = 0.1,  # Dropout率
        output_pos_dim: int = 3,  # 位置增量输出维度
        output_quat_dim: int = 4,  # 四元数增量输出维度
        output_scale_dim: int = 3,  # 尺度增量输出维度
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_pos_dim = output_pos_dim
        self.output_quat_dim = output_quat_dim
        self.output_scale_dim = output_scale_dim
        
        # 输入特征映射
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 双向LSTM
        self.bi_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        lstm_output_dim = hidden_dim * 2  # 双向LSTM输出维度
        
        # 位置增量预测分支
        self.pos_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_pos_dim)
        )
        
        # 四元数增量预测分支
        self.quat_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_quat_dim)
        )
        
        # 尺度增量预测分支
        self.scale_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_scale_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重使用Xavier初始化
                    nn.init.xavier_uniform_(param)
                else:
                    # 线性层权重使用He初始化
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 输入位置序列, shape: (batch_size, seq_len, 3)
            mask: 可选的掩码, shape: (batch_size, seq_len)
            
        Returns:
            Dict[str, torch.Tensor]: 包含增量更新的字典
                - delta_pos: 位置增量, shape: (batch_size, seq_len, 3)
                - delta_quat: 四元数增量, shape: (batch_size, seq_len, 4)
                - delta_scale: 尺度增量, shape: (batch_size, seq_len, 3)
        """
        batch_size, seq_len, _ = positions.shape
        
        # 输入特征映射
        x = self.input_projection(positions)  # (batch_size, seq_len, hidden_dim)
        
        # 双向LSTM
        lstm_out, _ = self.bi_lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # 如果有掩码，应用掩码
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            lstm_out = lstm_out * mask
        
        # 分别预测各个增量
        delta_pos = self.pos_head(lstm_out)  # (batch_size, seq_len, 3)
        delta_quat = self.quat_head(lstm_out)  # (batch_size, seq_len, 4)
        delta_scale = self.scale_head(lstm_out)  # (batch_size, seq_len, 3)
        
        # 对四元数增量进行归一化
        delta_quat = F.normalize(delta_quat, p=2, dim=-1)
        
        return {
            'delta_pos': delta_pos,
            'delta_quat': delta_quat,
            'delta_scale': delta_scale
        }


class GaussianSequenceRefiner:
    """高斯序列优化器 - 专注于推理功能"""
    
    def __init__(
        self,
        model: BiLSTMRefinement,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model.to(device)
        self.device = device
    
    def refine_sequences(self, initial_sequences: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        优化高斯序列 - 推理模式
        
        Args:
            initial_sequences: 初始高斯序列字典
                - positions: (num_frames, num_gaussians, 3)
                - orientations: (num_frames, num_gaussians, 4)
                - scales: (num_frames, num_gaussians, 3)
                - colors: (num_frames, num_gaussians, 3)
                - opacities: (num_frames, num_gaussians, 1)
        
        Returns:
            Dict[str, torch.Tensor]: 优化后的高斯序列
        """
        self.model.eval()
        
        with torch.no_grad():
            positions = initial_sequences['positions'].to(self.device)  # (num_frames, num_gaussians, 3)
            orientations = initial_sequences['orientations'].to(self.device)  # (num_frames, num_gaussians, 4)
            scales = initial_sequences['scales'].to(self.device)  # (num_frames, num_gaussians, 3)
            colors = initial_sequences['colors'].to(self.device)  # (num_frames, num_gaussians, 3)
            opacities = initial_sequences['opacities'].to(self.device)  # (num_frames, num_gaussians, 1)
            
            num_frames, num_gaussians, _ = positions.shape
            
            # 批量处理所有高斯点
            # 重新排列为 (num_gaussians, num_frames, 3) 用于LSTM处理
            positions_input = positions.transpose(0, 1)  # (num_gaussians, num_frames, 3)
            
            # 通过模型预测所有高斯点的增量
            deltas = self.model(positions_input)  # 批量处理
            
            # 应用位置增量
            refined_positions = positions_input + deltas['delta_pos']  # (num_gaussians, num_frames, 3)
            refined_positions = refined_positions.transpose(0, 1)  # (num_frames, num_gaussians, 3)
            
            # 应用四元数增量
            # 转换维度以匹配增量的形状
            orientations_reshaped = orientations.transpose(0, 1)  # (num_gaussians, num_frames, 4)
            refined_orientations = self._apply_quat_delta(orientations_reshaped, deltas['delta_quat'])
            refined_orientations = refined_orientations.transpose(0, 1)  # (num_frames, num_gaussians, 4)
            
            # 应用尺度增量
            scales_reshaped = scales.transpose(0, 1)  # (num_gaussians, num_frames, 3)
            refined_scales = scales_reshaped * torch.exp(deltas['delta_scale'])
            refined_scales = refined_scales.transpose(0, 1)  # (num_frames, num_gaussians, 3)
            
            # 组织结果
            refined_sequences = {
                'positions': refined_positions,  # (num_frames, num_gaussians, 3)
                'orientations': refined_orientations,  # (num_frames, num_gaussians, 4)
                'scales': refined_scales,  # (num_frames, num_gaussians, 3)
                'colors': colors,  # 颜色保持不变
                'opacities': opacities  # 透明度保持不变
            }
            
            return refined_sequences
    
    def _apply_quat_delta(self, original_quat: torch.Tensor, delta_quat: torch.Tensor) -> torch.Tensor:
        """
        应用四元数增量更新
        
        Args:
            original_quat: 原始四元数 (num_gaussians, num_frames, 4)
            delta_quat: 四元数增量 (num_gaussians, num_frames, 4)
            
        Returns:
            torch.Tensor: 更新后的四元数
        """
        # 使用四元数乘法应用增量
        # 这里简化为加法后归一化，实际应用中可能需要更复杂的四元数运算
        updated_quat = original_quat + delta_quat
        return F.normalize(updated_quat, p=2, dim=-1)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'output_pos_dim': self.model.output_pos_dim,
                'output_quat_dim': self.model.output_quat_dim,
                'output_scale_dim': self.model.output_scale_dim,
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: torch.device = None):
        """从检查点文件创建优化器"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 从配置创建模型
        model_config = checkpoint['model_config']
        model = BiLSTMRefinement(**model_config)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建优化器
        refiner = cls(model, device)
        
        print(f"Refiner loaded from {checkpoint_path}")
        return refiner


def main():
    """示例用法 - 演示推理功能"""
    print("=== BiLSTM Refinement 推理示例 ===")
    
    # 创建模型
    model = BiLSTMRefinement(
        input_dim=3,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    )
    
    # 创建优化器
    refiner = GaussianSequenceRefiner(model)
    
    # 创建模拟的高斯序列数据
    num_frames = 20
    num_gaussians = 40_000
    device = refiner.device
    
    initial_sequences = {
        'positions': torch.randn(num_frames, num_gaussians, 3, device=device),
        'orientations': torch.randn(num_frames, num_gaussians, 4, device=device),
        'scales': torch.ones(num_frames, num_gaussians, 3, device=device) * 0.1,
        'colors': torch.rand(num_frames, num_gaussians, 3, device=device),
        'opacities': torch.ones(num_frames, num_gaussians, 1, device=device) * 0.8
    }
    
    # 归一化四元数
    initial_sequences['orientations'] = F.normalize(initial_sequences['orientations'], p=2, dim=-1)
    
    print(f"Initial sequences shapes:")
    for key, value in initial_sequences.items():
        print(f"  {key}: {value.shape}")
    
    # 推理模式优化
    print("\n=== 推理模式优化 ===")
    refined_sequences = refiner.refine_sequences(initial_sequences)
    
    print(f"Refined sequences shapes:")
    for key, value in refined_sequences.items():
        print(f"  {key}: {value.shape}")
    
    # 计算位置变化
    position_diff = torch.mean((refined_sequences['positions'] - initial_sequences['positions']) ** 2)
    print(f"\nMean position change: {position_diff:.6f}")
    
    # 保存和加载模型示例
    print("\n=== 模型保存/加载示例 ===")
    model_path = "Gaussian_Sequence/output/temp_model.pt"
    refiner.save_model(model_path)
    
    # 从检查点加载
    refiner_loaded = GaussianSequenceRefiner.from_checkpoint(model_path)
    
    print("BiLSTM refinement model (inference-only) created successfully!")
    
    # 清理临时文件
    import os
    if os.path.exists(model_path):
        os.remove(model_path)


if __name__ == "__main__":
    main() 