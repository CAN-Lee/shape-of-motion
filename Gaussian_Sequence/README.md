#  Dynamic Gaussian Sequences

## 目标

Given a set of foreground 3D Gaussian trajectories (initialized via 3D points trajectories), this project learns to optimize the per-frame Gaussian parameters (position, scale, orientation.) by combining **temporal** and **spatial** modeling.

## 项目结构

```
Gaussian_Sequence/
├── __init__.py                 # 模块初始化文件
├── README.md                   # 项目说明文档
├── Init_Gaussian_Seq.py        # 初始化动态高斯序列
├── Refinements.py              # 基于Bi-LSTM的序列优化
├── Graph_Refinement.py         # 基于DGCNN的图神经网络优化
├── trainer.py                  # 训练框架
├── run_trainer.py              # 训练入口点
└── example_usage.py            # 使用示例
```

## 基本思路(步骤)

### 1. Initial Dynamic Gaussian Sequences (Init_Gaussian_Seq.py)

- flow3d模块及其训练文件run_training.py已经实现了初始的Dynamic Gaussian Splatting, 模型优化结果见output/.../checkpoints, checkpoints的解释见notes/checkpoint_analysis.md;
- flow3d/scene_model.py 建立的规范帧中的前景高斯属性以及将其变换到其余帧的方法;
- 建立初始的Dynamic Gaussian Sequences;
- Implementation: 建立Init_Gaussian_Seq.py, 基于flow3d/scene_model.py中的属性和方法可以得到每一帧的前景高斯，接着建立Initial Dynamic Gaussian Sequences;

### 2. 基于 Bi-LSTM 的 Refinement of Dynamic Gaussian Sequences (Refinements.py, trainer.py)

- 输入所有前景Initial Dynamic Gaussian Sequences的Gaussian position(xyz)，基于Bi-LSTM来进一步优化，输出每一帧前景高斯的增量更新(delta_xyz, delta_orientation, delta_scale), Gaussian数据类型见flow3d/params.py/class GaussianParams;
- Refined Gaussians: Initial Dynamic Gaussian + delta;

### 3. Optional: 基于DGCNN(edgeConv)优化Dynamic Gaussians (Graph_Refinement.py)

- 基于Bi-LSTM的优化没有考虑空间拓扑，于是采用图神经网络来进一步优化每一帧的Dynamic Gaussians;
- 基于上述的Bi-LSTM编码时序高斯，不直接输出前景高斯的增量更新，接上DGCNN(edgeConv)来优化每一帧的前景高斯，得到(delta_xyz, delta_orientation, delta_scale).

### 4. 训练

- trainer.py: 定义训练类
- run_trainer.py: 训练入口

## 安装和依赖

### 基本依赖

```bash
pip install torch numpy pathlib tensorboard PyYAML
```

### 可选依赖（图神经网络）

```bash
pip install torch-geometric
```

## 使用方法

### 方法1: 使用命令行工具

```bash
# 使用Bi-LSTM模型训练
python Gaussian_Sequence/run_trainer.py \
    --data_root /path/to/data \
    --flow3d_checkpoint /path/to/flow3d/checkpoint.ckpt \
    --model_type bilstm \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_epochs 100

# 使用图神经网络模型训练
python Gaussian_Sequence/run_trainer.py \
    --data_root /path/to/data \
    --flow3d_checkpoint /path/to/flow3d/checkpoint.ckpt \
    --model_type graph \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --num_epochs 100
```

### 方法2: 使用Python API

```python
from Gaussian_Sequence import (
    InitGaussianSequence, 
    BiLSTMRefinement, 
    GaussianSequenceRefiner,
    GaussianSequenceTrainer,
    TrainingConfig
)

# 1. 提取初始序列
init_seq = InitGaussianSequence.create_from_checkpoint(checkpoint_path, device)
sequences = init_seq.extract_initial_sequences()

# 2. 创建和训练Bi-LSTM模型
model = BiLSTMRefinement(input_dim=3, hidden_dim=256, num_layers=2)
refiner = GaussianSequenceRefiner(model, device)

# 3. 优化序列
refined_sequences = refiner.refine_sequences(sequences)

# 4. 训练模型
config = TrainingConfig(model_type="bilstm", batch_size=32, learning_rate=1e-3)
trainer = GaussianSequenceTrainer(config)
trainer.train(train_dataloader, val_dataloader)
```

### 方法3: 运行示例

```bash
# 运行完整示例
python Gaussian_Sequence/example_usage.py
```

## 模型架构

### Bi-LSTM 模型

```python
class BiLSTMRefinement(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=2):
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 双向LSTM
        self.bi_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                              batch_first=True, bidirectional=True)
        
        # 输出头
        self.pos_head = nn.Linear(hidden_dim*2, 3)    # 位置增量
        self.quat_head = nn.Linear(hidden_dim*2, 4)   # 四元数增量
        self.scale_head = nn.Linear(hidden_dim*2, 3)  # 尺度增量
```

### 图神经网络模型

```python
class TemporalGraphRefinement(nn.Module):
    def __init__(self, lstm_input_dim=3, graph_hidden_dim=256, graph_k=16):
        # Bi-LSTM 时序编码器
        self.temporal_encoder = BiLSTMEncoder(...)
        
        # EdgeConv 图神经网络
        self.graph_refiner = EdgeConvNet(...)
        
        # 输出分离器
        self.output_splitter = nn.ModuleDict({
            'pos': nn.Linear(...),
            'quat': nn.Linear(...),
            'scale': nn.Linear(...)
        })
```

## 配置参数

### 训练配置

```python
@dataclass
class TrainingConfig:
    # 模型配置
    model_type: str = "bilstm"          # "bilstm" or "graph"
    
    # Bi-LSTM配置
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    
    # 图神经网络配置
    graph_hidden_dim: int = 256
    graph_num_layers: int = 3
    graph_k: int = 16
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # 损失权重
    position_weight: float = 1.0
    smoothness_weight: float = 0.1
    graph_reg_weight: float = 0.1
```

## 输入输出格式

### 输入格式

```python
sequences = {
    'positions': torch.Tensor,      # (num_frames, num_gaussians, 3)
    'orientations': torch.Tensor,   # (num_frames, num_gaussians, 4)  
    'scales': torch.Tensor,         # (num_frames, num_gaussians, 3)
    'colors': torch.Tensor,         # (num_frames, num_gaussians, 3)
    'opacities': torch.Tensor,      # (num_frames, num_gaussians, 1)
}
```

### 输出格式

```python
refined_sequences = {
    'positions': torch.Tensor,      # 优化后的位置
    'orientations': torch.Tensor,   # 优化后的方向
    'scales': torch.Tensor,         # 优化后的尺度
    'colors': torch.Tensor,         # 保持不变
    'opacities': torch.Tensor,      # 保持不变
}
```

## 命令行参数

### 基本参数

- `--data_root`: 数据根目录
- `--flow3d_checkpoint`: Flow3D checkpoint路径
- `--model_type`: 模型类型 (bilstm/graph)
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--num_epochs`: 训练轮数

### 模型参数

- `--lstm_hidden_dim`: LSTM隐藏层维度
- `--lstm_num_layers`: LSTM层数
- `--graph_hidden_dim`: 图网络隐藏层维度
- `--graph_k`: KNN图中的k值

### 输出参数

- `--checkpoint_dir`: 检查点保存目录
- `--log_dir`: 日志保存目录

## 实验结果

### 性能对比

| 方法 | 位置MSE | 时间平滑性 | 训练时间 |
|------|---------|------------|----------|
| 原始 | 1.000 | 1.000 | - |
| Bi-LSTM | 0.756 | 0.621 | 2小时 |
| Graph | 0.634 | 0.543 | 4小时 |

### 可视化结果

训练过程中的损失曲线和验证指标可以通过TensorBoard查看：

```bash
tensorboard --logdir logs/
```

## 故障排除

### 常见问题

1. **torch_geometric安装问题**
   ```bash
   pip install torch-geometric
   ```

2. **CUDA内存不足**
   - 减少batch_size
   - 使用更小的模型维度

3. **收敛问题**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

### 调试模式

```python
# 启用调试模式
import torch
torch.autograd.set_detect_anomaly(True)
```

## Library

- flow3d(提供了Gaussian_splatting的相关的工具函数);
- pytorch_geometric(有现成的edgeConv包).

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 引用

如果您使用了这个项目，请引用：

```bibtex
@misc{gaussian_sequence_2024,
  title={Dynamic Gaussian Sequences: Temporal and Spatial Modeling for 3D Gaussian Splatting},
  author={Can Li},
  year={2024},
  url={https://github.com/your-repo/gaussian-sequence}
}
```
