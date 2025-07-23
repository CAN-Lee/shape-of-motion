"""
Gaussian_Sequence - Dynamic Gaussian Sequences优化模块

基于双向LSTM和图神经网络的动态高斯序列优化框架，用于时序和空间建模。

主要组件:
- Init_Gaussian_Seq: 初始化动态高斯序列
- Refinements: 基于Bi-LSTM的序列优化（支持批量处理和渲染损失）
- Graph_Refinement: 基于DGCNN的图神经网络优化
- trainer: 训练框架
- run_trainer: 训练入口点
- example_refinement: 完整的使用示例

关键更新（v1.1.0）:
- ✅ 批量处理所有高斯点，支持渲染损失计算
- ✅ 改进的训练API，支持多种损失函数
- ✅ 推理和训练模式分离
- ✅ 完整的使用示例和分析工具

用法示例:
    from Gaussian_Sequence import InitGaussianSequence, BiLSTMRefinement, GaussianSequenceRefiner
    
    # 提取初始序列
    init_seq = InitGaussianSequence.create_from_checkpoint(checkpoint_path, device)
    sequences = init_seq.extract_initial_sequences()
    
    # 创建和训练模型
    model = BiLSTMRefinement(input_dim=3, hidden_dim=256)
    refiner = GaussianSequenceRefiner(model, device)
    
    # 批量优化（推理模式）
    refined_sequences = refiner.refine_sequences_inference(sequences)
    
    # 训练模式（支持渲染损失）
    optimizer = torch.optim.Adam(model.parameters())
    losses = refiner.train_step(sequences, optimizer, scene_model, render_data)

作者: AI Assistant
日期: 2024
"""

from .Gaussian_Seq_model import GaussianSequenceModel
from Gaussian_Sequence.LSTM_Refinements import BiLSTMRefinement, GaussianSequenceRefiner
from .trainer import GaussianSequenceTrainer
from .run_trainer import TrainingConfig, GaussianSequenceDataset, create_dataloaders

# 可选的Graph模块（需要torch_geometric）
try:
    from .LSTM_Graph_Refinement import TemporalGraphRefinement, GraphGaussianRefiner
    from .EdgeConv_Example import DynamicEdgeConv, PointCloudClassifier, DynamicGraphDataset
    HAS_GRAPH_MODULE = True
except ImportError:
    HAS_GRAPH_MODULE = False

__version__ = "1.1.0"
__author__ = "AI Assistant"
__email__ = "ai.assistant@example.com"

__all__ = [
    # 初始化
    "GaussianSequenceModel",
    
    # Bi-LSTM优化
    "BiLSTMRefinement",
    "GaussianSequenceRefiner",
    
    # 训练相关
    "GaussianSequenceTrainer",
    "TrainingConfig",
    "GaussianSequenceDataset",
    "create_dataloaders",
    
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "HAS_GRAPH_MODULE"
]

# 如果有Graph模块，添加到__all__
if HAS_GRAPH_MODULE:
    __all__.extend([
        "TemporalGraphRefinement",
        "GraphGaussianRefiner",
        "DynamicEdgeConv",
        "PointCloudClassifier",
        "DynamicGraphDataset"
    ])


def get_version():
    """获取版本信息"""
    return __version__


def get_available_models():
    """获取可用的模型类型"""
    models = ["bilstm"]
    if HAS_GRAPH_MODULE:
        models.append("graph")
    return models


def check_dependencies():
    """检查依赖项"""
    dependencies = {
        "torch": True,
        "numpy": True,
        "pathlib": True,
        "torch_geometric": HAS_GRAPH_MODULE,
        "tensorboard": True,
        "yaml": True
    }
    
    missing = []
    for dep, available in dependencies.items():
        if not available:
            missing.append(dep)
    
    if missing:
        print(f"Warning: Missing dependencies: {missing}")
        if "torch_geometric" in missing:
            print("Install torch_geometric to use graph-based models:")
            print("pip install torch-geometric")
    
    return len(missing) == 0


# 模块初始化时检查依赖
# check_dependencies()

# print(f"Gaussian_Sequence v{__version__} loaded successfully!")
# print(f"Available models: {get_available_models()}")
# if not HAS_GRAPH_MODULE:
#     print("Note: Graph-based models require torch_geometric. Install with 'pip install torch-geometric'") 