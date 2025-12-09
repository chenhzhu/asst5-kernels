# Histogram Implementation Code Structure

## High-Level Architecture

The histogram implementation uses a **two-phase hierarchical reduction approach** to efficiently compute histograms for 2D input data `[length, num_channels]`:

1. **Phase 1**: Compute partial histograms in tiled blocks
2. **Phase 2**: Reduce partial histograms to final result

## Configuration

Firstly, we defined the basic **config** for the parallelization:

```cuda
// Config optimized for H100 (132 SMs)
dim3 block_dim(256, 1);
int channels_per_block = 64;
int grid_x = (num_channels + channels_per_block - 1) / channels_per_block; // 8

// H100 Optimized Chunks: Target grid_x * grid_y to be multiple of 132 SMs
int target_chunks = 33;
int chunk_size = (length + target_chunks - 1) / target_chunks;
if (chunk_size < 1) chunk_size = 1;
int grid_y = (length + chunk_size - 1) / chunk_size;

dim3 grid_dim(grid_x, grid_y);
```

As you can see, we defined the **block size** to be 256 threads, with **64 channels per block**, and **33 chunks** for the row dimension, which we tuned for optimized performance on H100 (264 blocks = 2 waves on 132 SMs).

## Phase 1: Partial Histogram Computation (`histogram_optimized_kernel`)

### Blocking Strategy

The input data has two dimensions: **rows (length)** and **channels (num_channels)**. The code implements a **2D blocking strategy** where:

- **Outer loop (channels dimension)**: The channels are blocked, with each CUDA thread block handling **64 channels**. This blocking is mapped to `blockIdx.x`.

- **Outer loop (rows dimension)**: The rows are blocked into chunks, with each CUDA thread block handling a **chunk of consecutive rows**. This blocking is mapped to `blockIdx.y`.

With these configurations, in the `histogram_optimized_kernel` function, we **block** the channels and rows:

```cuda
// 2. Determine Workload
int group_idx = blockIdx.x; 
int start_channel = group_idx * 64; // 64 channels per block

// Identify Y-chunk
int chunk_idx = blockIdx.y;
int row_start = chunk_idx * chunk_size;
int row_end = min(row_start + chunk_size, length);
```

Each thread block will process a `chunk_size × 64` chunk of the input data.

### Inner Loop Processing

For the inner loop, we process the rows within the assigned chunk. Each warp processes multiple rows using **8x loop unrolling** and **double buffering**:

```cuda
int effective_step = num_warps * 2;
int unroll_step_8x = effective_step * 8;
int r = row_start + warp_id * 2 + sub_row;

// DOUBLE BUFFERING + UNROLLING (8x for better ILP on H100)
unsigned int v0, v1, v2, v3, v4, v5, v6, v7;

if (has_full_batch_8x) {
    // Prefetch first batch
    v0 = ptr_base[r * row_stride_u32];
    v1 = ptr_base[(r + effective_step) * row_stride_u32];
    // ... v2 through v7
    r += unroll_step_8x;
    
    while (r + effective_step * 7 < row_end) {
        // Prefetch next batch while processing current
        unsigned int n0 = ptr_base[r * row_stride_u32];
        unsigned int n1 = ptr_base[(r + effective_step) * row_stride_u32];
        
        // Process v0-v1
        atomicAdd(&h0[v0 & 0xFF], 1);
        atomicAdd(&h1[(v0 >> 8) & 0xFF], 1);
        atomicAdd(&h2[(v0 >> 16) & 0xFF], 1);
        atomicAdd(&h3[v0 >> 24], 1);
        // ... process v1 through v7
        
        // Update buffers
        v0 = n0; v1 = n1; v2 = n2; v3 = n3;
        v4 = n4; v5 = n5; v6 = n6; v7 = n7;
        r += unroll_step_8x;
    }
}
```

Each iteration processes 8 rows simultaneously, with double buffering to overlap memory loads with computation, which hides memory latency and improves GPU performance.

### Shared Memory Accumulation

After chunking, we use **shared memory** to accumulate histogram counts locally:

```cuda
// Shared Memory: [64 channels][258 bins]
extern __shared__ int s_hist[];

// 1. Initialize Shared Memory
for (int i = tid; i < 64 * 258; i += blockDim.x) {
    s_hist[i] = 0;
}
__syncthreads();

// During processing, accumulate in shared memory
atomicAdd(&h0[v0 & 0xFF], 1);
atomicAdd(&h1[(v0 >> 8) & 0xFF], 1);
atomicAdd(&h2[(v0 >> 16) & 0xFF], 1);
atomicAdd(&h3[v0 >> 24], 1);
```

The shared memory layout uses **257 bins per channel** (256 bins + 1 padding) to avoid bank conflicts. Each thread processes 4 consecutive channels via vectorized 32-bit loads, extracting 4 bytes from each word.

### Flush to Global Memory

After processing all rows in the chunk, we flush the partial histogram to global memory:

```cuda
// 3. Flush (Privatized Store with Warp-Level Coalescing)
int total_elements = num_channels * num_bins;
int chunk_offset = chunk_idx * total_elements;

for (int i = tid; i < 64 * 256; i += blockDim.x) {
    int ch_local = i / 256;
    int bin = i % 256;
    
    int s_addr = (ch_local * 257) + (ch_local >> 2) + bin;
    int count = s_hist[s_addr];
    
    if (count > 0) {
        int global_ch = start_channel + ch_local;
        int global_idx = chunk_offset + global_ch * num_bins + bin;
        partial_output[global_idx] = count;
    }
}
```

## Phase 2: Reduction (`histogram_reduce_kernel`)

### Structure

After computing partial histograms, we reduce them to the final result. Each thread handles one output element (one channel-bin pair):

```cuda
__global__ void histogram_reduce_kernel(
    const int* __restrict__ partial_histograms,
    int* __restrict__ output,
    int num_elements, // num_channels * num_bins
    int grid_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        int sum = 0;
        int offset = idx;
        
        // Unrolled accumulation for better ILP (4-way unroll)
        int i = 0;
        for (; i + 3 < grid_y; i += 4) {
            sum += partial_histograms[offset];
            offset += num_elements;
            sum += partial_histograms[offset];
            offset += num_elements;
            sum += partial_histograms[offset];
            offset += num_elements;
            sum += partial_histograms[offset];
            offset += num_elements;
        }
        
        // Handle remainder
        for (; i < grid_y; ++i) {
            sum += partial_histograms[offset];
            offset += num_elements;
        }
        
        output[idx] = sum;
    }
}
```

The reduction sums across the `grid_y` dimension (chunks) to combine partial histograms, using **4-way loop unrolling** for better ILP.

### Input/Output

- **Input**: `[grid_y, num_channels, num_bins]` partial histograms (flattened)
- **Output**: `[num_channels, num_bins]` final histogram

## Key Optimizations

1. **Hierarchical reduction**: Avoids global atomic contention by first accumulating in shared memory, then reducing partial results
2. **Vectorized memory access**: Reads 4 channels at once as 32-bit words
3. **Bank conflict avoidance**: Padding in shared memory layout (257 bins per channel instead of 256)
4. **ILP optimization**: 8x unrolling in Phase 1, 4x unrolling in Phase 2
5. **Double buffering**: Overlaps memory loads with computation in Phase 1
6. **H100-specific tuning**: Grid dimensions chosen to maximize SM utilization (264 blocks = 2 waves on 132 SMs)

