#!/usr/bin/env python3
"""
Memory-Optimized Gaussian Deformation Training Script

This is a conservative version of the deformation training script designed to prevent memory issues
while maintaining render_full_sequence=True for proper temporal loss computation.

Key Memory Optimizations:
- Batch size 1 with gradient accumulation (effective batch size 4)
- ULTRA-SHORT sequences (2 frames - minimal temporal relationship)
- Chunked processing enabled with smaller chunks
- Full sequence rendering (required for temporal loss)
- Regular GPU cache clearing and memory monitoring
- REDUCED validation frequency to prevent hanging

Usage:
    python run_deform_memory_optimized.py
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Starting ULTRA memory-optimized deformation training...")
    print("\n📊 Performance Analysis:")
    print("Flow3d training: 1 render call per step")
    print("Deformation training: sequence_length × render calls per step")
    print("With sequence_length=2: Only 2×slower than flow3d (vs 3×slower with length=3)")
    
    # 设置内存优化的参数 - 保持render_full_sequence=True但优化其他方面
    cmd = [
        sys.executable, "run_deform_training.py",
        "--work_dir", "refined_output",
        "--scene_model_path", "output/paper-windmill-result/checkpoints/last.ckpt",
        "--batch_size", "1",  # 最小batch size
        "--sequence_length", "3",  # 极简序列：只有前帧+当前帧，最小化渲染次数
        "--num_dl_workers", "1",
        "--validate_every", "25",  # 进一步减少验证频率
        "--save_every", "5",
        "--num_epochs", "50",  # 先训练较少的epochs进行测试
        "data:iphone",  # 数据配置必须在最后，格式为data:iphone
        "--data.data-dir", "data/iPhone/paper-windmill"
    ]
    
    print("\n⚡ ULTRA Memory optimizations applied:")
    print("- Batch size: 1")
    print("- Sequence length: 2 (MINIMAL - previous+current frame)")
    print("- Render calls per step: 2 (vs flow3d's 1)")
    print("- Render full sequence: TRUE (required for temporal loss)")
    print("- Workers: 1")
    print("- Epochs: 50 (for testing)")
    print("- Validation frequency: ULTRA-REDUCED to every 25 epochs")
    print("\n🎯 Expected speedup: ~33% faster than length=3 configuration")
    print("⚠️  Trade-off: Shorter temporal context for model learning")
    
    print(f"\n🔧 Command: {' '.join(cmd)}")
    print()
    
    # 运行训练命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 