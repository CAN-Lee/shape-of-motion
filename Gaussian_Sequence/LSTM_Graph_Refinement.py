"""
基于DGCNN(edgeConv)优化Dynamic Gaussians (Graph_Refinement.py)

基于Bi-LSTM的优化没有考虑空间拓扑，于是采用图神经网络来进一步优化每一帧的Dynamic Gaussians。
基于上述的Bi-LSTM编码时序高斯，不直接输出前景高斯的增量更新，接上DGCNN(edgeConv)来优化每一帧的前景高斯。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool, knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from Gaussian_Sequence.LSTM_Refinements import BiLSTMRefinement
import math

class TemporalGraphRefinement(nn.Module):
    """时序图优化网络：结合Bi-LSTM和DGCNN"""
    
    def __init__(
        self,
        # Bi-LSTM参数
        lstm_input_dim: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
        # DGCNN参数
        graph_input_dim: int = 512,  # 来自Bi-LSTM的特征维度
        graph_hidden_dim: int = 256,
        graph_num_layers: int = 3,
        graph_k: int = 16,
        graph_dropout: float = 0.1,
        # 输出参数
        output_pos_dim: int = 3,
        output_quat_dim: int = 4,
        output_scale_dim: int = 3,
    ):
        super().__init__()
        
        self.lstm_input_dim = lstm_input_dim  # 保存LSTM输入维度
        self.lstm_hidden_dim = lstm_hidden_dim
        self.output_pos_dim = output_pos_dim
        self.output_quat_dim = output_quat_dim
        self.output_scale_dim = output_scale_dim
        
        # Bi-LSTM编码器（不输出增量，只输出特征）
        self.temporal_encoder = nn.Module()
        self.temporal_encoder.input_projection = nn.Linear(lstm_input_dim, lstm_hidden_dim)
        self.temporal_encoder.bi_lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0
        )
        
        # 动态图神经网络 (DynamicEdgeConv)
        self.graph_k = graph_k
        graph_output_dim = output_pos_dim + output_quat_dim + output_scale_dim
        
        # 图网络的实际输入维度 = 位置维度(3) + LSTM特征维度(双向LSTM输出)
        # 注意：这里位置维度固定为3，因为图网络需要原始3D位置来构建邻接关系
        actual_graph_input_dim = 3 + graph_input_dim  # 3 for position, graph_input_dim for LSTM features
        
        # 构建维度列表：[input_dim, hidden_dim, hidden_dim, ..., output_dim]
        dims = [actual_graph_input_dim] + [graph_hidden_dim] * (graph_num_layers - 1) + [graph_output_dim]
        
        # 构建DynamicEdgeConv层
        self.graph_layers = nn.ModuleList()
        for i in range(graph_num_layers):
            self.graph_layers.append(
                DynamicEdgeConv(
                    nn.Sequential(
                        nn.Linear(dims[i] * 2, graph_hidden_dim),  # DynamicEdgeConv输入维度需要×2
                        nn.ReLU(),
                        nn.Dropout(graph_dropout),
                        nn.Linear(graph_hidden_dim, dims[i + 1])  # 输出到下一层的维度
                    ),
                    k=graph_k,
                    aggr='max'
                )
            )
        
        # 输出分离器：将最终的图特征分离为不同的增量
        self.output_splitter = nn.ModuleDict({
            'pos': nn.Linear(graph_output_dim, output_pos_dim),
            'quat': nn.Linear(graph_output_dim, output_quat_dim),
            'scale': nn.Linear(graph_output_dim, output_scale_dim)
        })

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重
        
        LSTM权重初始化策略:
        - 输入门和遗忘门的偏置初始化为较大的正值(1.0)，帮助在训练初期保留更多信息
        - 输出门的偏置初始化为0
        - 权重使用正交初始化，有助于减缓梯度消失/爆炸
        
        Graph网络初始化策略:
        - DynamicEdgeConv中的线性层使用Kaiming初始化
        - 输出分离器使用Xavier初始化
        """
        # 1. 时序编码器初始化
        # Input projection
        nn.init.kaiming_normal_(self.temporal_encoder.input_projection.weight, nonlinearity='relu')
        nn.init.zeros_(self.temporal_encoder.input_projection.bias)
        
        # Bi-LSTM
        for name, param in self.temporal_encoder.bi_lstm.named_parameters():
            if 'weight_ih' in name:  # 输入权重
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:  # 隐藏状态权重
                nn.init.orthogonal_(param.data)
            elif 'bias_ih' in name:  # 输入偏置
                param.data.fill_(0)
                # f_gate是遗忘门，i_gate是输入门
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # 遗忘门偏置设为1
                param.data[:n//4].fill_(1.0)      # 输入门偏置设为1
            elif 'bias_hh' in name:  # 隐藏状态偏置
                param.data.fill_(0)
        
        # 2. 图网络层初始化
        for layer in self.graph_layers:
            # DynamicEdgeConv的MLP中的线性层
            for name, module in layer.nn.named_modules():
                if isinstance(module, nn.Linear):
                    # 使用Kaiming初始化，适合ReLU激活函数
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(module.bias, -bound, bound)
        
        # 3. 输出分离器初始化
        for name, module in self.output_splitter.items():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，因为这些层后面没有非线性激活
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, positions: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            positions: 位置+时间编码序列, shape: (batch_size, seq_len, lstm_input_dim) 
                      如果lstm_input_dim=19，则前3维是位置，后16维是时间编码
            batch_indices: 批次索引, shape: (num_nodes,) - 当输入是 (num_nodes, seq_len, 3) 时使用
            
        Returns:
            Dict[str, torch.Tensor]: 包含增量更新的字典
        """
        batch_size, seq_len, input_dim = positions.shape
        
        # 分离位置和时间编码（如果存在）
        if input_dim > 3:
            # 输入包含时间编码，前3维是位置，后面是时间编码
            frame_positions_3d = positions[:, :, :3]  # (batch_size, seq_len, 3)
            # 使用完整输入进行LSTM处理
            lstm_input = positions  # (batch_size, seq_len, lstm_input_dim)
        else:
            # 输入只有位置信息
            frame_positions_3d = positions  # (batch_size, seq_len, 3)
            lstm_input = positions  # (batch_size, seq_len, 3)
        
        # 1. 时序编码 (Bi-LSTM)
        # 输入特征映射
        x = self.temporal_encoder.input_projection(lstm_input)  # (batch_size, seq_len, hidden_dim)
        
        # 双向LSTM
        lstm_out, _ = self.temporal_encoder.bi_lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # 2. 空间优化 (DGCNN) - 并行处理所有时间步
        # 重新组织数据以实现时间步的并行处理
        # batch_size, seq_len, hidden_dim = lstm_out.shape
        lstm_features_flat = lstm_out.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, hidden_dim)
        positions_flat = frame_positions_3d.reshape(batch_size * seq_len, 3)    # (batch_size * seq_len, 3)

        # 创建批次索引，用于区分不同的时间步和样本
        if batch_indices is None:
            # 对于时间序列数据，所有时间步的节点应该属于同一个图
            # 这样可以让不同时间步之间的节点进行交互
            # batch_indices = torch.zeros(batch_size * seq_len, dtype=torch.long, device=lstm_out.device)
            # 为每个时间步创建正确的batch索引
            batch_indices = torch.arange(batch_size, device=positions.device).repeat_interleave(seq_len)
        else:
            # 扩展现有的批次索引以覆盖所有时间步
            batch_indices = batch_indices.repeat_interleave(seq_len)

        # 初始化图输出
        graph_output = lstm_features_flat

        # 通过动态图神经网络层 - 现在可以并行处理所有时间步
        for i, layer in enumerate(self.graph_layers):
            if i == 0:
                # 第一层结合位置和特征
                combined_input = torch.cat([positions_flat, graph_output], dim=-1)
                graph_output = layer(combined_input, batch_indices)
            else:
                graph_output = layer(graph_output, batch_indices)
            
            # 不在最后一层应用ReLU
            if i < len(self.graph_layers) - 1:
                graph_output = F.relu(graph_output)

        # 4. 分离不同的输出
        delta_pos = self.output_splitter['pos'](graph_output)
        delta_quat = self.output_splitter['quat'](graph_output)
        delta_scale = self.output_splitter['scale'](graph_output)
        
        # 归一化四元数
        delta_quat = F.normalize(delta_quat, p=2, dim=-1)
        
        # 5. 重塑回序列形式
        delta_pos = delta_pos.reshape(batch_size, seq_len, self.output_pos_dim)
        delta_quat = delta_quat.reshape(batch_size, seq_len, self.output_quat_dim)
        delta_scale = delta_scale.reshape(batch_size, seq_len, self.output_scale_dim)
        
        return {
            'delta_pos': delta_pos,
            'delta_quat': delta_quat,
            'delta_scale': delta_scale
        }


class GaussianRefiner:
    """基于图神经网络的高斯序列优化器"""
    
    def __init__(
        self,
        model: TemporalGraphRefinement,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model.to(device)
        self.device = device
    
    def refine_sequences(self, initial_sequences: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        使用图神经网络优化高斯序列
        
        Args:
            initial_sequences: 初始高斯序列字典
            
        Returns:
            Dict[str, torch.Tensor]: 优化后的高斯序列
        """
        self.model.eval()
        
        positions = initial_sequences['positions'].to(self.device)  # (num_frames, num_gaussians, 3)
        orientations = initial_sequences['orientations'].to(self.device)
        scales = initial_sequences['scales'].to(self.device)
        # 颜色和不透明度保持不变，不需要在序列中处理
        
        # 转换维度顺序以适应模型输入
        positions_input = positions.transpose(0, 1)  # (num_gaussians, num_frames, 3)
        
        refined_positions = []
        refined_orientations = []
        refined_scales = []
        
        with torch.no_grad():
            # 分批处理以避免内存问题
            batch_size = min(1000, positions_input.shape[0])  # 每批处理的高斯点数
            
            for i in range(0, positions_input.shape[0], batch_size):
                end_idx = min(i + batch_size, positions_input.shape[0])
                batch_positions = positions_input[i:end_idx]  # (batch_size, num_frames, 3)
                
                # 前向传播
                deltas = self.model(batch_positions)
                
                # 应用增量更新
                batch_refined_pos = batch_positions + deltas['delta_pos']
                refined_positions.append(batch_refined_pos)
                
                # 处理四元数和尺度
                batch_orientations = orientations[:, i:end_idx, :].transpose(0, 1)  # (batch_size, num_frames, 4)
                batch_scales = scales[:, i:end_idx, :].transpose(0, 1)  # (batch_size, num_frames, 3)
                
                # 应用增量
                batch_refined_quat = self._apply_quat_delta(batch_orientations, deltas['delta_quat'])
                batch_refined_scale = batch_scales * torch.exp(deltas['delta_scale'])
                
                refined_orientations.append(batch_refined_quat)
                refined_scales.append(batch_refined_scale)
        
        # 合并结果
        refined_positions = torch.cat(refined_positions, dim=0).transpose(0, 1)  # (num_frames, num_gaussians, 3)
        refined_orientations = torch.cat(refined_orientations, dim=0).transpose(0, 1)
        refined_scales = torch.cat(refined_scales, dim=0).transpose(0, 1)
        
        refined_sequences = {
            'positions': refined_positions,
            'orientations': refined_orientations,
            'scales': refined_scales,
            # 颜色和不透明度保持不变，不需要在refined_sequences中包含
        }
        
        return refined_sequences
    
    def _apply_quat_delta(self, original_quat: torch.Tensor, delta_quat: torch.Tensor) -> torch.Tensor:
        """应用四元数增量更新"""
        updated_quat = original_quat + delta_quat
        return F.normalize(updated_quat, p=2, dim=-1)
    

    
    def save_model(self, path: str):
        """保存模型（仅推理用）"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, path)
        print(f"Graph refinement model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型（仅推理用）"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 设置为评估模式
        print(f"Graph refinement model loaded from {path}")


def main():
    """测试Graph模型的基本功能"""
    print("Testing Graph Refinement Model...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = TemporalGraphRefinement(
        lstm_input_dim=3,
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        graph_input_dim=512,
        graph_hidden_dim=256,
        graph_num_layers=3,
        graph_k=16
    )
    
    # 创建推理器
    refiner = GaussianRefiner(model, device)
    
    # 示例数据
    batch_size = 40_000
    seq_len = 10
    input_dim = 3
    
    # 创建随机输入数据并移动到正确设备
    positions = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    print(f"Input shape: {positions.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    with torch.no_grad():
        outputs = model(positions)
        
        print("\nModel output shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
    
    # 测试推理器
    print("\nTesting refiner with dummy sequences...")
    dummy_sequences = {
        'positions': torch.randn(seq_len, batch_size, 3).to(device),
        'orientations': torch.nn.functional.normalize(torch.randn(seq_len, batch_size, 4), p=2, dim=-1).to(device),
        'scales': torch.ones(seq_len, batch_size, 3).to(device) * 0.1,
        'colors': torch.rand(seq_len, batch_size, 3).to(device),
        'opacities': torch.rand(seq_len, batch_size, 1).to(device)
    }
    
    try:
        refined_sequences = refiner.refine_sequences(dummy_sequences)
        print(f"✓ Refiner test successful!")
        print(f"  Refined positions shape: {refined_sequences['positions'].shape}")
    except Exception as e:
        print(f"✗ Refiner test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nGraph refinement model test completed!")


if __name__ == "__main__":
    main() 