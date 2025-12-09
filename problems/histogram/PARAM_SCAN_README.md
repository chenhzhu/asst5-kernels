# 参数扫描使用指南

我已经修改了代码以支持参数扫描。以下是使用方法：

## 修改内容

1. **CUDA 代码已更新** (`template.cu`):
   - `histogram_kernel` 函数现在接受可配置参数
   - Kernel 函数接受 `channels_per_block` 参数
   - 所有硬编码值已改为可配置

2. **创建了三个脚本**:
   - `quick_test.py`: 快速测试当前配置
   - `param_scan.py`: 参数扫描框架（需要手动修改代码）
   - `auto_param_scan.py`: 自动参数扫描（自动修改和重新编译）

## 快速开始

### 方法 1: 手动修改参数（推荐，最简单）

1. **修改 `submission.cu` 中的参数**（约第 283-338 行）:

```cuda
torch::Tensor histogram_kernel(
    torch::Tensor data,
    int num_bins,
    int block_size = 256,              // 修改这里
    int channels_per_block = 64,       // 修改这里
    int target_chunks = 33,             // 修改这里
    int shared_mem_carveout = 75,      // 修改这里
    int reduce_block_size = 256        // 修改这里
) {
```

2. **重新编译**:
```bash
python wrap_cuda_submission.py <你的SUNetID>
```

3. **快速测试**:
```bash
python quick_test.py
```

### 方法 2: 使用快速测试脚本

1. 修改 `submission.cu` 中的参数值
2. 重新编译
3. 运行 `python quick_test.py` 查看性能

### 方法 3: 批量测试（需要手动修改）

1. 编辑 `param_scan.py`，修改 `generate_configs()` 函数中的配置列表
2. 对于每个配置：
   - 手动修改 `submission.cu` 中的参数
   - 重新编译
   - 运行测试

## 推荐测试的参数组合

基于性能分析，建议按以下顺序测试：

### 1. 测试 Shared Memory Carveout（最简单，可能最有效）

```cuda
// 测试值: 50, 75, 100
int shared_mem_carveout = 50;  // 或 75, 100
```

**为什么**: L1 cache hit rate 是 0%，增加 L1 cache 可能有助于缓存输入数据。

### 2. 测试 Target Chunks

```cuda
// 测试值: 16, 32, 33, 64, 128
int target_chunks = 16;  // 或 32, 64, 128
```

**为什么**: 改变总 block 数量，可能提高 SM 利用率（当前只有 34%）。

### 3. 测试 Channels Per Block

```cuda
// 测试值: 32, 48, 64, 96, 128
int channels_per_block = 32;  // 或 48, 96, 128
```

**注意**: 需要同时修改 kernel 中的硬编码值（如果还有的话）。

### 4. 测试 Block Size

```cuda
// 测试值: 128, 256, 384, 512
dim3 block_dim(128, 1);  // 或 256, 384, 512
```

## 测试流程示例

```bash
# 1. 修改参数（例如：shared_mem_carveout = 50）
# 编辑 submission.cu，修改第 337 行

# 2. 重新编译
python wrap_cuda_submission.py <SUNetID>

# 3. 测试
python quick_test.py

# 4. 记录结果
# 查看 benchmark_result.txt 或手动记录

# 5. 尝试下一个参数组合
# 重复步骤 1-4
```

## 参数说明

| 参数 | 当前值 | 测试范围 | 影响 |
|------|--------|----------|------|
| `block_size` | 256 | 128, 256, 384, 512 | 每个 block 的线程数 |
| `channels_per_block` | 64 | 32, 48, 64, 96, 128 | 每个 block 处理的通道数 |
| `target_chunks` | 33 | 16, 32, 33, 64, 128 | 行方向的 chunk 数量 |
| `shared_mem_carveout` | 75 | 50, 75, 100 | Shared memory vs L1 cache 分配 |
| `reduce_block_size` | 256 | 128, 256, 512 | Reduction kernel 的 block size |

## 预期改进

基于 profiling 结果：

- **Shared Memory Carveout**: 可能提高 L1 cache hit rate（当前 0%）
- **减少 Channels Per Block**: 可能提高 SM 利用率（当前 34%）
- **增加 Chunks**: 可能改善负载均衡

## 注意事项

1. **每次修改参数后必须重新编译**
2. **确保参数组合有效**（例如 channels_per_block 不能太大，否则 shared memory 不够）
3. **记录每次测试的结果**，方便比较
4. **测试时使用相同的输入数据**，确保公平比较

## 结果分析

测试完成后，比较不同配置的：
- **Mean time**: 平均执行时间
- **Std dev**: 标准差（稳定性）
- **Min/Max time**: 最坏/最好情况

选择 **Mean time 最小** 且 **Std dev 较小**（稳定）的配置。

## 快速参考

```bash
# 修改参数 → 重新编译 → 测试
vim submission.cu                    # 修改参数
python wrap_cuda_submission.py <ID>   # 重新编译
python quick_test.py                  # 测试性能
```

祝优化顺利！

