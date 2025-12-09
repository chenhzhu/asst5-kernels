# Simple Parameter Tuning Optimizations

基于 profiling 结果，以下是一些**仅通过调整参数**就能尝试的简单优化方法，无需修改核心算法逻辑。

## 1. 调整 Shared Memory Carveout（最简单，可能最有效）

### 当前设置
```cuda
cudaFuncSetAttribute(
    histogram_optimized_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    100  // 100% shared memory, 0% L1 cache
);
```

### 优化建议
由于 L1 cache hit rate 是 0%，可以尝试**减少 shared memory carveout，增加 L1 cache**：

```cuda
// 尝试选项 1: 75% shared memory, 25% L1 cache
cudaFuncSetAttribute(
    histogram_optimized_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    75
);

// 尝试选项 2: 50% shared memory, 50% L1 cache
cudaFuncSetAttribute(
    histogram_optimized_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    50
);
```

**原理**：当前所有数据都从 DRAM 读取，增加 L1 cache 可能有助于缓存输入数据，提高访问速度。

**测试值**：`[50, 75, 100]`（当前是100）

---

## 2. 调整 Channels Per Block

### 当前设置
```cuda
int channels_per_block = 64;
```

### 优化建议
调整每个 block 处理的通道数，影响：
- Shared memory 使用量
- Block 数量（grid_x）
- 内存访问模式

```cuda
// 选项 1: 减少到 32 channels（减少 shared memory，增加 blocks）
int channels_per_block = 32;  // grid_x = 16, shared_mem = 32 * 258 * 4 = 33KB

// 选项 2: 增加到 128 channels（增加 shared memory，减少 blocks）
int channels_per_block = 128;  // grid_x = 4, shared_mem = 128 * 258 * 4 = 132KB
```

**注意**：需要同时更新 kernel 中的硬编码值：
```cuda
// 在 histogram_optimized_kernel 中
int start_channel = group_idx * channels_per_block;  // 需要改为参数
// 以及
for (int i = tid; i < channels_per_block * 258; i += blockDim.x) {  // 需要改为参数
```

**测试值**：`[32, 48, 64, 96, 128]`（当前是64）

**权衡**：
- **更少 channels**：更多 blocks，更好的 SM 利用率，但可能增加 reduction 开销
- **更多 channels**：更少 blocks，但 shared memory 压力更大

---

## 3. 调整 Chunk Size / Target Chunks

### 当前设置
```cuda
int target_chunks = 33;
int chunk_size = (length + target_chunks - 1) / target_chunks;
```

### 优化建议
调整行方向的 chunk 数量，影响：
- grid_y 大小
- 总 block 数量（grid_x * grid_y）
- Reduction 阶段的开销

```cuda
// 选项 1: 减少 chunks（更大的 chunk_size）
int target_chunks = 16;  // grid_y = 16, 总 blocks = 8 * 16 = 128
// 或
int target_chunks = 17;  // grid_y = 17, 总 blocks = 8 * 17 = 136

// 选项 2: 增加 chunks（更小的 chunk_size）
int target_chunks = 64;  // grid_y = 64, 总 blocks = 8 * 64 = 512
```

**测试值**：`[16, 17, 32, 33, 64, 128]`（当前是33）

**权衡**：
- **更少 chunks**：更少的 reduction 开销，但可能 SM 利用率不足
- **更多 chunks**：更好的负载均衡，但 reduction 阶段开销更大

**H100 优化**：目标是 `grid_x * grid_y` 接近 132 的倍数
- 当前：8 * 33 = 264（2 waves）
- 尝试：8 * 16 = 128（1 wave），8 * 17 = 136（略多于1 wave），8 * 64 = 512（约4 waves）

---

## 4. 调整 Block Size (Threads Per Block)

### 当前设置
```cuda
dim3 block_dim(256, 1);
```

### 优化建议
调整每个 block 的线程数，影响：
- Warp 数量
- 内存访问的并行度
- Register 压力

```cuda
// 选项 1: 减少到 128 threads（4 warps）
dim3 block_dim(128, 1);

// 选项 2: 增加到 512 threads（16 warps，需要检查 register 限制）
dim3 block_dim(512, 1);
```

**注意**：需要确保是 32 的倍数（warp size）

**测试值**：`[128, 256, 384, 512]`（当前是256）

**权衡**：
- **更少 threads**：更多 blocks，更好的 SM 利用率，但可能减少并行度
- **更多 threads**：更少 blocks，但可能受 register 限制

---

## 5. 调整 Reduction Kernel Block Size

### 当前设置
```cuda
int reduce_block_size = 256;
```

### 优化建议
调整 reduction kernel 的 block size：

```cuda
// 选项 1: 减少到 128
int reduce_block_size = 128;

// 选项 2: 增加到 512
int reduce_block_size = 512;
```

**测试值**：`[128, 256, 512]`（当前是256）

---

## 6. 组合优化策略

### 策略 A：最大化 SM 利用率
```cuda
dim3 block_dim(128, 1);           // 减少 threads per block
int channels_per_block = 32;       // 减少 channels per block
int target_chunks = 64;             // 增加 chunks
// 结果：更多 blocks，更好的 SM 利用率
```

### 策略 B：最大化 Cache 利用
```cuda
cudaFuncSetAttribute(
    histogram_optimized_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    50  // 50% shared memory, 50% L1 cache
);
int target_chunks = 16;  // 更大的 chunks，可能提高 cache 重用
```

### 策略 C：平衡 Shared Memory 和 Blocks
```cuda
int channels_per_block = 48;        // 中等值
int target_chunks = 32;             // 中等值
dim3 block_dim(256, 1);             // 保持当前值
```

---

## 快速测试脚本建议

创建一个简单的参数扫描：

```python
# 伪代码
configs = [
    {"channels_per_block": 32, "target_chunks": 64, "block_size": 128, "carveout": 50},
    {"channels_per_block": 32, "target_chunks": 64, "block_size": 256, "carveout": 75},
    {"channels_per_block": 48, "target_chunks": 32, "block_size": 256, "carveout": 50},
    {"channels_per_block": 64, "target_chunks": 16, "block_size": 256, "carveout": 50},
    {"channels_per_block": 64, "target_chunks": 33, "block_size": 256, "carveout": 75},  # 当前配置的变体
    # ... 更多组合
]

for config in configs:
    # 修改代码参数
    # 运行 benchmark
    # 记录性能
```

---

## 预期效果

基于 profiling 结果，最可能有效的调整：

1. **Shared Memory Carveout (50-75%)**：可能提高 L1 cache hit rate
2. **减少 Channels Per Block (32-48)**：增加 blocks，提高 SM 利用率
3. **增加 Chunks (64-128)**：更好的负载均衡，但需要权衡 reduction 开销

**注意**：这些是**启发式优化**，实际效果需要通过 profiling 验证。建议：
1. 先单独测试每个参数
2. 找到最佳的单参数值
3. 然后组合测试多个参数

---

## 实现注意事项

如果要让这些参数可调，需要：

1. **将硬编码值改为参数**：
   - `channels_per_block` 需要传递给 kernel
   - Kernel 中的 `64 * 258` 需要改为 `channels_per_block * 258`
   - `start_channel = group_idx * 64` 需要改为 `group_idx * channels_per_block`

2. **或者创建多个 kernel 版本**：
   - 为不同的 `channels_per_block` 值创建专门的 kernel
   - 使用模板或宏来生成不同版本

3. **最简单的实现**：
   - 先手动修改代码中的硬编码值
   - 测试不同配置
   - 找到最佳配置后固定下来

