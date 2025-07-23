"""
run_trainer.py - 训练入口

用于启动Dynamic Gaussian Sequences的训练过程。
支持命令行参数配置和多种训练模式。
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from Gaussian_Sequence.trainer import GaussianSequenceTrainer
from Gaussian_Sequence.Gaussian_Seq_model import GaussianSequenceModel


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_type: str = "bilstm"  # "bilstm" or "graph"
    
    # Bi-LSTM配置
    lstm_input_dim: int = 3
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    
    # Graph配置
    graph_input_dim: int = 512
    graph_hidden_dim: int = 256
    graph_num_layers: int = 3
    graph_k: int = 16
    graph_dropout: float = 0.1
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 1e-4
    
    # RGB损失权重配置 - 参考flow3d
    rgb_l1_weight: float = 0.8
    rgb_ssim_weight: float = 0.2
    use_masked_loss: bool = True
    loss_quantile: float = 0.98
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 100
    validate_interval: int = 50
    
    # 路径配置
    data_root: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GaussianSequenceDataset(Dataset):
    """高斯序列数据集"""
    
    def __init__(
        self,
        sequences_path: str,
        sequence_length: int = 50,
        stride: int = 10,
        augment: bool = True
    ):
        """
        Args:
            sequences_path: 序列数据路径
            sequence_length: 序列长度
            stride: 滑动窗口步长
            augment: 是否进行数据增强
        """
        self.sequences = torch.load(sequences_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment = augment
        
        # 提取位置信息
        self.positions = self.sequences['positions']  # (num_frames, num_gaussians, 3)
        self.orientations = self.sequences['orientations']  # (num_frames, num_gaussians, 4)
        self.scales = self.sequences['scales']  # (num_frames, num_gaussians, 3)
        self.colors = self.sequences.get('colors', torch.rand_like(self.positions))
        self.opacities = self.sequences.get('opacities', torch.ones(self.positions.shape[0], self.positions.shape[1], 1))
        
        self.num_frames, self.num_gaussians, _ = self.positions.shape
        
        # 创建训练样本索引
        self.samples = []
        for start_frame in range(0, self.num_frames - sequence_length + 1, stride):
            self.samples.append(start_frame)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_frame = self.samples[idx]
        end_frame = start_frame + self.sequence_length
        
        # 提取序列片段
        positions = self.positions[start_frame:end_frame]  # (seq_len, num_gaussians, 3)
        orientations = self.orientations[start_frame:end_frame]  # (seq_len, num_gaussians, 4)
        scales = self.scales[start_frame:end_frame]  # (seq_len, num_gaussians, 3)
        colors = self.colors[start_frame:end_frame]  # (seq_len, num_gaussians, 3)
        opacities = self.opacities[start_frame:end_frame]  # (seq_len, num_gaussians, 1)
        
        # 数据增强
        if self.augment:
            # 添加位置噪声
            positions = positions + torch.randn_like(positions) * 0.01
            # 添加尺度噪声
            scales = scales * (1 + torch.randn_like(scales) * 0.05)
            scales = torch.clamp(scales, min=1e-6)  # 保证尺度为正
        
        return {
            'positions': positions,
            'orientations': orientations,
            'scales': scales,
            'colors': colors,
            'opacities': opacities,
            'start_frame': start_frame
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练Dynamic Gaussian Sequences模型")
    
    # 数据配置
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据根目录路径')
    parser.add_argument('--sequences_path', type=str, default=None,
                        help='预处理的序列数据路径。如果未提供，将从flow3d checkpoint生成')
    parser.add_argument('--flow3d_checkpoint', type=str, default=None,
                        help='Flow3D checkpoint路径，用于生成初始序列')
    
    # 模型配置
    parser.add_argument('--model_type', type=str, default='bilstm',
                        choices=['bilstm', 'graph'],
                        help='模型类型: bilstm 或 graph')
    
    # Bi-LSTM配置
    parser.add_argument('--lstm_hidden_dim', type=int, default=256,
                        help='LSTM隐藏层维度')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='LSTM层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.1,
                        help='LSTM dropout率')
    
    # Graph配置
    parser.add_argument('--graph_hidden_dim', type=int, default=256,
                        help='图神经网络隐藏层维度')
    parser.add_argument('--graph_num_layers', type=int, default=3,
                        help='图神经网络层数')
    parser.add_argument('--graph_k', type=int, default=16,
                        help='KNN图中的k值')
    parser.add_argument('--graph_dropout', type=float, default=0.1,
                        help='图神经网络dropout率')
    
    # 训练配置
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # RGB损失权重配置
    parser.add_argument('--rgb_l1_weight', type=float, default=0.8,
                        help='RGB L1损失权重')
    parser.add_argument('--rgb_ssim_weight', type=float, default=0.2,
                        help='RGB SSIM损失权重')
    parser.add_argument('--use_masked_loss', action='store_true',
                        help='是否使用掩码损失')
    parser.add_argument('--loss_quantile', type=float, default=0.98,
                        help='损失分位数阈值')
    
    # 日志和保存
    parser.add_argument('--log_interval', type=int, default=10,
                        help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='模型保存间隔')
    parser.add_argument('--validate_interval', type=int, default=50,
                        help='验证间隔')
    
    # 输出目录
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='日志保存目录')
    
    # 设备配置
    parser.add_argument('--device', type=str, default='auto',
                        help='设备类型: auto, cpu, cuda')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    
    # 数据分割
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='训练数据比例')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证数据比例')
    
    # 序列参数
    parser.add_argument('--sequence_length', type=int, default=50,
                        help='序列长度')
    parser.add_argument('--stride', type=int, default=10,
                        help='滑动窗口步长')
    
    return parser.parse_args()


def setup_device(device_arg: str, gpu_id: int):
    """设置设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def generate_sequences_from_checkpoint(flow3d_checkpoint: str, output_path: str, device: torch.device):
    """从Flow3D checkpoint生成初始序列"""
    print(f"Generating sequences from Flow3D checkpoint: {flow3d_checkpoint}")
    
    # 创建初始序列提取器
    init_seq = InitGaussianSequence.create_from_checkpoint(flow3d_checkpoint, device)
    
    # 提取序列
    sequences = init_seq.extract_initial_sequences()
    
    # 保存序列
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    init_seq.save_sequences(sequences, output_path)
    
    print(f"Sequences saved to: {output_path}")
    return sequences


def create_dataloaders(
    sequences_path: str,
    config: TrainingConfig,
    train_split: float = 0.8,
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 创建数据集
    full_dataset = GaussianSequenceDataset(
        sequences_path=sequences_path,
        sequence_length=50,
        stride=10,
        augment=True
    )
    
    # 分割数据集
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader


def save_config(config: TrainingConfig, config_path: str):
    """保存配置文件"""
    with open(config_path, 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")


def load_config(config_path: str) -> TrainingConfig:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def save_checkpoint(trainer: GaussianSequenceTrainer, checkpoint_path: str, is_best: bool = False, is_final: bool = False):
    """保存检查点"""
    checkpoint = {
        'epoch': trainer.epoch,
        'global_step': trainer.global_step,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'best_loss': trainer.best_loss,
        'config': trainer.config
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        print(f"Best model saved at epoch {trainer.epoch}")
    elif is_final:
        print(f"Final model saved at epoch {trainer.epoch}")


def load_checkpoint(trainer: GaussianSequenceTrainer, checkpoint_path: str):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    trainer.epoch = checkpoint['epoch']
    trainer.global_step = checkpoint['global_step']
    trainer.best_loss = checkpoint['best_loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming training from epoch {trainer.epoch}")


def create_config_from_args(args) -> TrainingConfig:
    """从命令行参数创建训练配置"""
    device = setup_device(args.device, args.gpu_id)
    
    # 创建输出目录
    checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_type)
    log_dir = os.path.join(args.log_dir, args.model_type)
    
    config = TrainingConfig(
        # 模型配置
        model_type=args.model_type,
        
        # Bi-LSTM配置
        lstm_input_dim=3,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=args.lstm_dropout,
        
        # Graph配置
        graph_input_dim=args.lstm_hidden_dim * 2,  # 双向LSTM输出
        graph_hidden_dim=args.graph_hidden_dim,
        graph_num_layers=args.graph_num_layers,
        graph_k=args.graph_k,
        graph_dropout=args.graph_dropout,
        
        # 训练配置
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        
        # RGB损失权重配置
        rgb_l1_weight=args.rgb_l1_weight,
        rgb_ssim_weight=args.rgb_ssim_weight,
        use_masked_loss=args.use_masked_loss,
        loss_quantile=args.loss_quantile,
        
        # 日志和保存
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        validate_interval=args.validate_interval,
        
        # 路径配置
        data_root=args.data_root,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        
        # 设备
        device=str(device)
    )
    
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 从配置文件加载配置（如果提供）
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        # 可以用命令行参数覆盖配置文件中的某些参数
        if hasattr(args, 'num_epochs') and args.num_epochs != 100:
            config.num_epochs = args.num_epochs
        if hasattr(args, 'batch_size') and args.batch_size != 32:
            config.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate != 1e-3:
            config.learning_rate = args.learning_rate
    else:
        # 从命令行参数创建配置
        config = create_config_from_args(args)
    
    # 创建输出目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 处理序列数据
    sequences_path = args.sequences_path
    if sequences_path is None:
        # 从Flow3D checkpoint生成序列
        if args.flow3d_checkpoint is None:
            raise ValueError("必须提供 --sequences_path 或 --flow3d_checkpoint")
        
        sequences_path = os.path.join(config.data_root, "initial_sequences.pt")
        generate_sequences_from_checkpoint(
            args.flow3d_checkpoint, 
            sequences_path, 
            torch.device(config.device)
        )
    
    # 检查序列文件是否存在
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(f"序列文件不存在: {sequences_path}")
    
    # 创建训练器
    trainer = GaussianSequenceTrainer(config)
    
    print(f"Starting training with {config.model_type} model")
    print(f"Device: {config.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    
    # 保存配置
    config_path = os.path.join(config.checkpoint_dir, "config.yaml")
    save_config(config, config_path)
    
    # 恢复训练（如果提供）
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        load_checkpoint(trainer, args.resume)
    
    # 创建数据加载器
    print("Creating data loaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        sequences_path, 
        config,
        train_split=args.train_split,
        val_split=args.val_split
    )
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # 开始训练
    print("Starting training...")
    
    for epoch in range(config.num_epochs):
        trainer.epoch = epoch
        
        # 训练一个epoch
        train_losses = trainer.train_epoch(train_dataloader)
        
        # 验证
        if val_dataloader is not None and epoch % config.validate_interval == 0:
            val_losses = trainer.validate(val_dataloader)
            
            # 保存最佳模型
            if val_losses['rgb_loss'] < trainer.best_loss:
                trainer.best_loss = val_losses['rgb_loss']
                best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                save_checkpoint(trainer, best_path, is_best=True)
        
        # 保存检查点
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(trainer, checkpoint_path)
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"Train Loss: {train_losses['total_loss']:.6f}")
        if val_dataloader is not None and epoch % config.validate_interval == 0:
            print(f"Val RGB Loss: {val_losses['rgb_loss']:.6f}")
        print(f"Learning Rate: {trainer.scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
    
    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    save_checkpoint(trainer, final_path, is_final=True)
    
    # 关闭tensorboard writer
    trainer.writer.close()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main() 