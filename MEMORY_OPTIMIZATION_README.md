# Gaussian Deformation Training Memory Optimization

## 问题分析

原始训练脚本在第一个epoch后卡住，主要原因是GPU内存不足导致进程被系统杀死。

## 内存优化策略

### 1. 减少批次大小和序列长度
- `batch_size: 1`（最小可行批次）
- `sequence_length: 3`（最小有意义序列：前一帧、当前帧、后一帧）

### 2. 梯度累积
- `gradient_accumulation_steps: 4`
- 模拟有效批次大小为4，保持训练稳定性

### 3. 分块处理
- `use_chunked_processing: True`
- `chunk_size: 300`（更小的块以适应全序列渲染）

### 4. 内存管理
- 每10步清理GPU缓存
- 每50步监控内存使用
- 显式删除临时渲染结果
- 减少worker进程数

## 使用方法

### 方法1：使用优化后的默认配置
```bash
python run_deform_training.py --work_dir refined_output nvidia
```

### 方法2：使用专用的内存优化启动脚本
```bash
python run_deform_memory_optimized.py
```

## 内存使用估算

基于新配置的内存使用：
- Batch size: 1
- Sequence length: 3  
- 每批次渲染：1 × 3 = 3张图片（vs 原来的 4 × 10 = 40张）
- 内存减少约：3/40 = 7.5%的原始使用量

## 监控和调试

训练过程会显示：
- GPU内存使用情况
- 序列帧差异（确保使用真实序列而非重复帧）
- 梯度累积状态

如果仍然出现内存问题，可以进一步：
1. 减少`chunk_size`到200或更少
2. 减少`sequence_length`到2（前帧+当前帧）
3. 增加`gradient_accumulation_steps`到8或更多 