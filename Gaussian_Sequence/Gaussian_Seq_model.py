"""
Initial Dynamic Gaussian Sequences (Init_Gaussian_Seq.py)

基于flow3d/scene_model.py中的属性和方法，建立初始的Dynamic Gaussian Sequences。
从训练好的flow3d模型中提取每一帧的前景高斯参数。
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import os
import sys
import torch.nn.functional as F
import roma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow3d.scene_model import SceneModel
from flow3d.params import GaussianParams
from Gaussian_Sequence.LSTM_Refinements import GaussianSequenceRefiner, BiLSTMRefinement


class GaussianSequenceModel(nn.Module):
    """初始化动态高斯序列"""
    
    def __init__(self, scene_model: SceneModel, device: torch.device = None):
        """
        Args:
            scene_model: 训练好的SceneModel实例
            device: 设备，如果为None则自动检测
        """
        super().__init__()
        self.scene_model = scene_model
        self.num_frames = scene_model.num_frames
        self.num_fg_gaussians = scene_model.num_fg_gaussians
        self.num_bg_gaussians = scene_model.num_bg_gaussians
        self.has_bg = scene_model.has_bg
        
        # 获取设备
        if device is None:
            # 尝试从模型参数获取设备
            try:
                self.device = next(self.scene_model.parameters()).device
            except StopIteration:
                # 如果模型没有参数，使用CPU
                self.device = torch.device('cpu')
        else:
            self.device = device
    
    def compute_poses_bg(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            means: (G, B, 3)
            quats: (G, B, 4)
        """
        assert self.scene_model.bg is not None
        return self.scene_model.bg.params["means"], self.scene_model.bg.get_quats()
    
    def compute_poses_fg(
        self, ts: torch.Tensor | None, inds: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            ts: (T,)
            inds: (G,)
        :returns means: (G, B, 3), quats: (G, B, 4)
        """
        means = self.scene_model.fg.params["means"]  # (G, 3)
        quats = self.scene_model.fg.get_quats()  # (G, 4)
        if inds is not None:
            means = means[inds]
            quats = quats[inds]
        if ts is not None:
            transfms = self.scene_model.compute_transforms(ts, inds)  # (G, B, 3, 4)
            means = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(means, (0, 1), value=1.0),
            )
            quats = roma.quat_xyzw_to_wxyz(
                (
                    roma.quat_product(
                        roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                        roma.quat_wxyz_to_xyzw(quats[:, None]),
                    )
                )
            )
            quats = F.normalize(quats, p=2, dim=-1)
        else:
            means = means[:, None]
            quats = quats[:, None]
        return means, quats
    
    def get_initial_Gaussian_sequences(self, frame_indices: torch.Tensor = None) -> dict:
        """
        从SceneModel中提取初始的Dynamic Gaussian Sequences
        
        Args:
            frame_indices: 指定的帧索引，如果为None则使用所有帧
            
        Returns:
            dict: 包含所有帧的高斯参数序列
                - positions: (num_frames, num_gaussians, 3) 位置序列
                - orientations: (num_frames, num_gaussians, 4) 四元数序列
                - scales: (num_frames, num_gaussians, 3) 尺度序列
                - colors: (num_frames, num_gaussians, 3) 颜色序列
                - opacities: (num_frames, num_gaussians, 1) 透明度序列
        """
        if frame_indices is None:
            frame_indices = torch.arange(self.num_frames, device=self.device)
        
        # 存储所有帧的高斯参数
        all_positions = []
        all_orientations = []
        all_scales = []
        all_colors = []
        all_opacities = []
        
        self.scene_model.eval()
        with torch.no_grad():
            for t in frame_indices:
                # 计算前景高斯在第t帧的位置和方向
                positions, orientations = self.scene_model.compute_poses_fg(t.unsqueeze(0))
                # positions: (num_gaussians, 1, 3), orientations: (num_gaussians, 1, 4)
                
                # 获取尺度、颜色和透明度（这些在不同帧中保持不变）
                scales = self.scene_model.fg.get_scales()  # (num_gaussians, 3)
                colors = self.scene_model.fg.get_colors()  # (num_gaussians, 3)
                opacities = self.scene_model.fg.get_opacities()  # (num_gaussians, 1)
                
                # 去除多余的维度
                positions = positions.squeeze(1)  # (num_gaussians, 3)
                orientations = orientations.squeeze(1)  # (num_gaussians, 4)
                
                all_positions.append(positions)
                all_orientations.append(orientations)
                all_scales.append(scales)
                all_colors.append(colors)
                all_opacities.append(opacities)
        
        # 堆叠所有帧的参数
        sequences = {
            'positions': torch.stack(all_positions, dim=0),  # (num_frames, num_gaussians, 3)
            'orientations': torch.stack(all_orientations, dim=0),  # (num_frames, num_gaussians, 4)
            'scales': torch.stack(all_scales, dim=0),  # (num_frames, num_gaussians, 3)
            'colors': torch.stack(all_colors, dim=0),  # (num_frames, num_gaussians, 3)
            'opacities': torch.stack(all_opacities, dim=0),  # (num_frames, num_gaussians, 1)
        }
        
        return sequences
    
    def get_initial_trajectory(self, gaussian_idx: int) -> torch.Tensor:
        """
        获取特定高斯点的轨迹
        
        Args:
            gaussian_idx: 高斯点索引
            
        Returns:
            torch.Tensor: 形状为(num_frames, 3)的位置轨迹
        """
        frame_indices = torch.arange(self.num_frames, device=self.device)
        trajectory = []
        
        self.scene_model.eval()
        with torch.no_grad():
            for t in frame_indices:
                positions, _ = self.scene_model.compute_poses_fg(t.unsqueeze(0))
                trajectory.append(positions[gaussian_idx, 0])  # 取第gaussian_idx个高斯点的位置
        
        return torch.stack(trajectory, dim=0)
    
    def get_refined_Gaussian_sequences(self, frame_indices: torch.Tensor = None, 
                                     refinement_model_path: str = None) -> dict:
        """
        获取细化后的高斯序列
        
        Args:
            frame_indices: 指定的帧索引，如果为None则使用所有帧
            refinement_model_path: 预训练的refinement模型路径，如果为None则使用默认模型
            
        Returns:
            dict: 包含细化后的高斯参数序列
                - positions: (num_frames, num_gaussians, 3) 优化后的位置序列
                - orientations: (num_frames, num_gaussians, 4) 优化后的四元数序列
                - scales: (num_frames, num_gaussians, 3) 优化后的尺度序列
                - colors: (num_frames, num_gaussians, 3) 颜色序列（保持不变）
                - opacities: (num_frames, num_gaussians, 1) 透明度序列（保持不变）
        """
        pass
    
    def save_sequences(self, sequences: dict, save_path: str):
        """
        保存高斯序列到文件
        
        Args:
            sequences: 高斯序列字典
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sequences, save_path)
        print(f"Saved initial Gaussian sequences to {save_path}")
    
    def load_sequences(self, load_path: str) -> dict:
        """
        从文件加载高斯序列
        
        Args:
            load_path: 加载路径
            
        Returns:
            dict: 高斯序列字典
        """
        sequences = torch.load(load_path)
        print(f"Loaded initial Gaussian sequences from {load_path}")
        return sequences
    
    @staticmethod
    def init_from_checkpoint(checkpoint_path: str, device: torch.device) -> 'GaussianSequenceModel':
        """
        从flow3d训练的checkpoint创建GaussianSequenceModel实例
        
        Args:
            checkpoint_path: checkpoint文件路径
            device: 设备
            
        Returns:
            GaussianSequenceModel实例
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"]
        scene_model = SceneModel.init_from_state_dict(state_dict)
        scene_model = scene_model.to(device)
        
        return GaussianSequenceModel(scene_model, device)


def main():
    """示例用法"""
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从checkpoint加载
    checkpoint_path = "output/paper-windmill-result/checkpoints/last.ckpt"
    init_seq = GaussianSequenceModel.init_from_checkpoint(checkpoint_path, device)
    
    # 提取初始序列
    sequences = init_seq.get_initial_Gaussian_sequences()
    
    # 保存序列
    init_seq.save_sequences(sequences, "Gaussian_Sequence/output/initial_gaussian_sequences.pt")
    
    # 获取特定高斯点的轨迹
    trajectory = init_seq.get_initial_trajectory(gaussian_idx=0)
    print(f"Trajectory shape: {trajectory.shape}")
    
    print("Initial Gaussian sequences created successfully!")
    
    # 使用refinement模型获取精细化后的序列
    print("\n=== 使用 Refinements.py 获取精细化后的高斯序列 ===")
    refined_sequences = init_seq.get_refined_Gaussian_sequences()
    
    # 保存精细化后的序列
    init_seq.save_sequences(refined_sequences, "Gaussian_Sequence/output/refined_gaussian_sequences.pt")
    
    print("Refined Gaussian sequences created successfully!")


if __name__ == "__main__":
    main() 