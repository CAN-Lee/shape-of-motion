"""
EdgeConv示例 - 支持动态图大小 (EdgeConv_Example.py)

本示例展示了如何使用EdgeConv处理节点数不固定的图数据，
包括动态图构建、批处理、以及完整的训练/推理流程。

EdgeConv是Dynamic Graph CNN中的核心操作，特别适合处理点云数据。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from torch_geometric.nn import EdgeConv, global_max_pool, global_mean_pool, knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch



class DynamicEdgeConv(nn.Module):
    """
    支持动态图大小的EdgeConv网络
    
    特点：
    - 支持不同节点数的图
    - 动态k-NN图构建
    - 批处理不同大小的图
    - 多层EdgeConv堆叠
    """
    
    def __init__(
        self,
        input_dim: int = 3,      # 输入特征维度
        hidden_dims: List[int] = [64, 128, 256],  # 隐藏层维度列表
        output_dim: int = 32,    # 输出特征维度
        k: int = 16,             # k-NN中的k值
        dropout: float = 0.1,    # Dropout率
        use_global_pool: bool = True,  # 是否使用全局池化
        pool_method: str = 'max',      # 池化方法：'max', 'mean', 'sum'
    ):
        super().__init__()
        
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.k = k
        self.use_global_pool = use_global_pool
        self.pool_method = pool_method
        
        # 构建EdgeConv层
        self.edge_convs = nn.ModuleList()
        
        # 输入维度
        current_dim = input_dim
        
        # 创建多层EdgeConv
        for hidden_dim in hidden_dims:
            self.edge_convs.append(
                EdgeConv(
                    nn.Sequential(
                        nn.Linear(current_dim * 2, hidden_dim),  # EdgeConv输入是2倍特征
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    )
                )
            )
            current_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(current_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        # 如果使用全局池化，添加最终分类/回归层
        if use_global_pool:
            self.final_layer = nn.Linear(output_dim, output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征, shape: (num_nodes, input_dim)
            pos: 节点位置, shape: (num_nodes, 3) - 用于构建图结构
            batch: 批次索引, shape: (num_nodes,) - 指示每个节点属于哪个图
            
        Returns:
            torch.Tensor: 输出特征
                - 如果use_global_pool=True: (batch_size, output_dim)
                - 如果use_global_pool=False: (num_nodes, output_dim)
        """
        # 构建动态k-NN图
        edge_index = knn_graph(pos, self.k, batch=batch, loop=False)
        
        # 通过EdgeConv层
        for edge_conv in self.edge_convs:
            x = edge_conv(x, edge_index)
        
        # 输出层
        x = self.output_layer(x)
        
        # 全局池化
        if self.use_global_pool:
            if self.pool_method == 'max':
                x = global_max_pool(x, batch)
            elif self.pool_method == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pool_method == 'sum':
                x = global_max_pool(x, batch) + global_mean_pool(x, batch)
            else:
                raise ValueError(f"Unknown pool method: {self.pool_method}")
            
            # 最终层
            x = self.final_layer(x)
        
        return x


class PointCloudClassifier(nn.Module):
    """基于EdgeConv的点云分类器示例"""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_dim: int = 3,
        hidden_dims: List[int] = [64, 128, 256],
        k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = DynamicEdgeConv(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=256,
            k=k,
            dropout=dropout,
            use_global_pool=True,
            pool_method='max'
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征, shape: (num_nodes, input_dim)
            pos: 节点位置, shape: (num_nodes, 3)
            batch: 批次索引, shape: (num_nodes,)
            
        Returns:
            torch.Tensor: 分类结果, shape: (batch_size, num_classes)
        """
        features = self.backbone(x, pos, batch)
        return self.classifier(features)


class DynamicGraphDataset:
    """动态图数据集 - 支持不同节点数的图"""
    
    def __init__(self, num_samples: int = 1000, min_nodes: int = 100, max_nodes: int = 2000):
        self.num_samples = num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        # 生成随机图数据
        self.data_list = []
        self.labels = []
        
        for i in range(num_samples):
            # 随机节点数
            num_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # 生成随机点云数据
            pos = torch.randn(num_nodes, 3)
            
            # 生成特征（可以是位置本身或其他特征）
            features = pos + torch.randn(num_nodes, 3) * 0.1
            
            # 生成标签（这里简单用节点数范围作为类别）
            if num_nodes < 500:
                label = 0  # 小图
            elif num_nodes < 1000:
                label = 1  # 中图
            else:
                label = 2  # 大图
            
            self.data_list.append((features, pos))
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        features, pos = self.data_list[idx]
        label = self.labels[idx]
        return features, pos, label
    
    def create_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建批次数据
        
        Args:
            indices: 样本索引列表
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - features: (total_nodes, feature_dim)
                - pos: (total_nodes, 3)
                - batch: (total_nodes,) - 批次索引
                - labels: (batch_size,)
        """
        batch_features = []
        batch_pos = []
        batch_indices = []
        batch_labels = []
        
        for i, idx in enumerate(indices):
            features, pos, label = self[idx]
            
            batch_features.append(features)
            batch_pos.append(pos)
            batch_indices.append(torch.full((features.shape[0],), i, dtype=torch.long))
            batch_labels.append(label)
        
        # 拼接所有数据
        batch_features = torch.cat(batch_features, dim=0)
        batch_pos = torch.cat(batch_pos, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        
        return batch_features, batch_pos, batch_indices, batch_labels


def train_epoch(model: nn.Module, dataset: DynamicGraphDataset, optimizer: torch.optim.Optimizer, 
                batch_size: int = 4, device: torch.device = torch.device('cpu')) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 随机采样批次
    num_batches = len(dataset) // batch_size
    
    for batch_idx in range(num_batches):
        # 随机选择样本
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        
        # 创建批次
        features, pos, batch_indices, labels = dataset.create_batch(indices)
        
        # 移动到设备
        features = features.to(device)
        pos = pos.to(device)
        batch_indices = batch_indices.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(features, pos, batch_indices)
        
        # 计算损失
        loss = F.cross_entropy(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / num_batches
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total
    }


def test_model(model: nn.Module, dataset: DynamicGraphDataset, 
               batch_size: int = 4, device: torch.device = torch.device('cpu')) -> Dict[str, float]:
    """测试模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        # 测试所有数据
        num_batches = len(dataset) // batch_size
        
        for batch_idx in range(num_batches):
            # 顺序选择样本
            start_idx = batch_idx * batch_size
            indices = list(range(start_idx, min(start_idx + batch_size, len(dataset))))
            
            # 创建批次
            features, pos, batch_indices, labels = dataset.create_batch(indices)
            
            # 移动到设备
            features = features.to(device)
            pos = pos.to(device)
            batch_indices = batch_indices.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features, pos, batch_indices)
            
            # 计算损失
            loss = F.cross_entropy(outputs, labels)
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / num_batches
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total
    }


def visualize_graph_sizes(dataset: DynamicGraphDataset, save_path: Optional[str] = None):
    """可视化图的大小分布"""
    node_counts = []
    labels = []
    
    for i in range(len(dataset)):
        features, pos, label = dataset[i]
        node_counts.append(features.shape[0])
        labels.append(label)
    
    plt.figure(figsize=(12, 4))
    
    # 子图1: 节点数分布
    plt.subplot(1, 2, 1)
    plt.hist(node_counts, bins=50, alpha=0.7, color='skyblue')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Graph Sizes')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 标签分布
    plt.subplot(1, 2, 2)
    label_counts = [labels.count(i) for i in range(3)]
    plt.bar(['Small (<500)', 'Medium (500-1000)', 'Large (>1000)'], label_counts, color=['lightcoral', 'lightsalmon', 'lightgreen'])
    plt.xlabel('Graph Size Category')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    """主函数 - 完整的EdgeConv示例"""
    print("=== EdgeConv动态图示例 ===")
    
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n=== 创建数据集 ===")
    train_dataset = DynamicGraphDataset(num_samples=800, min_nodes=100, max_nodes=2000)
    test_dataset = DynamicGraphDataset(num_samples=200, min_nodes=100, max_nodes=2000)
    
    print(f"训练数据集: {len(train_dataset)} 个样本")
    print(f"测试数据集: {len(test_dataset)} 个样本")
    
    # 可视化数据集
    print("\n=== 数据集可视化 ===")
    output_dir = Path("Gaussian_Sequence/output")
    output_dir.mkdir(exist_ok=True)
    visualize_graph_sizes(train_dataset, output_dir / "graph_sizes.png")
    
    # 创建模型
    print("\n=== 创建模型 ===")
    model = PointCloudClassifier(
        num_classes=3,
        input_dim=3,
        hidden_dims=[64, 128, 256],
        k=16,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试单个批次
    print("\n=== 测试单个批次 ===")
    batch_indices = [0, 1, 2, 3]
    features, pos, batch_indices_tensor, labels = train_dataset.create_batch(batch_indices)
    
    print(f"批次特征形状: {features.shape}")
    print(f"批次位置形状: {pos.shape}")
    print(f"批次索引形状: {batch_indices_tensor.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 测试前向传播
    features = features.to(device)
    pos = pos.to(device)
    batch_indices_tensor = batch_indices_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(features, pos, batch_indices_tensor)
        print(f"输出形状: {outputs.shape}")
    
    # 训练模型
    print("\n=== 训练模型 ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    batch_size = 4
    
    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_dataset, optimizer, batch_size, device)
        
        # 测试
        test_metrics = test_model(model, test_dataset, batch_size, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  测试 - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.2f}%")
    
    # 保存模型
    print("\n=== 保存模型 ===")
    model_path = output_dir / "edgeconv_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': 3,
            'input_dim': 3,
            'hidden_dims': [64, 128, 256],
            'k': 16,
            'dropout': 0.1
        }
    }, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 展示EdgeConv的特点
    print("\n=== EdgeConv特点总结 ===")
    print("1. 动态图构建：每次前向传播都重新构建k-NN图")
    print("2. 支持不同节点数：可以处理任意大小的图")
    print("3. 边特征学习：EdgeConv学习边上的特征关系")
    print("4. 平移不变性：对点云的平移变换具有不变性")
    print("5. 批处理支持：可以批量处理不同大小的图")
    
    print("\nEdgeConv示例运行完成！")


if __name__ == "__main__":
    main() 