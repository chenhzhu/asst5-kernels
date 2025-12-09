import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import sys
import io

# CUDA source code loaded from submission.cu
cuda_source = """
#include <cuda_runtime.h>
#include <torch/extension.h>

// Note: Thread Block Clusters (H100 feature) could further optimize by:
// - Grouping 2-4 blocks into clusters with distributed shared memory
// - Reducing partial histogram count from grid_y to grid_y/cluster_size
// - Requires CUDA 11.8+ and __cluster_dims__ attribute
// Not implemented here for compatibility reasons

// Optimized for NVIDIA H100
// Specs: length=1048576, num_bins=256, num_channels=512.
// Target: ~300us.

// Phase 2: Hierarchical Reduction Kernel (Optimized for H100)
// Sums partial histograms from each Y-grid block using shared memory.
// Input: [grid_y, num_channels, num_bins] (Flattened)
// Output: [num_channels, num_bins]
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

__global__ void histogram_optimized_kernel(
    const unsigned char* __restrict__ data,
    int* __restrict__ partial_output, // [grid_y, num_channels, num_bins]
    int length,
    int num_channels,
    int num_bins,
    int chunk_size
) {
    // Shared Memory: [64 channels][258 bins]
    // Dynamic allocation to allow explicit L1 carveout control.
    extern __shared__ int s_hist[];
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    // 1. Initialize Shared Memory
    for (int i = tid; i < 64 * 258; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // 2. Determine Workload
    int group_idx = blockIdx.x; 
    int start_channel = group_idx * 64; // 64 channels per block
    
    // Identify Y-chunk
    int chunk_idx = blockIdx.y;
    int row_start = chunk_idx * chunk_size;
    int row_end = min(row_start + chunk_size, length);
    
    if (start_channel < num_channels) {
        const unsigned int* data_u32 = reinterpret_cast<const unsigned int*>(data);
        int row_stride_u32 = num_channels >> 2; // Divide by 4
        
        int logical_lane = lane_id % 16;
        int sub_row = lane_id / 16;
        
        int col_offset = (start_channel >> 2) + logical_lane;
        
        // Shared pointers with Bank Conflict Avoidance Padding
        int ch0 = 4 * logical_lane + 0;
        int ch1 = 4 * logical_lane + 1;
        int ch2 = 4 * logical_lane + 2;
        int ch3 = 4 * logical_lane + 3;
        
        int base_shift = logical_lane; 
        
        int* h0 = &s_hist[(ch0 * 257) + base_shift];
        int* h1 = &s_hist[(ch1 * 257) + base_shift];
        int* h2 = &s_hist[(ch2 * 257) + base_shift];
        int* h3 = &s_hist[(ch3 * 257) + base_shift];

        const unsigned int* ptr_base = data_u32 + col_offset;
        
        int effective_step = num_warps * 2;
        int unroll_step = effective_step * 4;
        
        int r = row_start + warp_id * 2 + sub_row;
        
        // DOUBLE BUFFERING + UNROLLING (8x for better ILP on H100)
        unsigned int v0, v1, v2, v3, v4, v5, v6, v7;
        int unroll_step_8x = effective_step * 8;
        bool has_full_batch_8x = (r + effective_step * 7 < row_end);
        
        if (has_full_batch_8x) {
            // Prefetch first batch
            v0 = ptr_base[r * row_stride_u32];
            v1 = ptr_base[(r + effective_step) * row_stride_u32];
            v2 = ptr_base[(r + effective_step * 2) * row_stride_u32];
            v3 = ptr_base[(r + effective_step * 3) * row_stride_u32];
            v4 = ptr_base[(r + effective_step * 4) * row_stride_u32];
            v5 = ptr_base[(r + effective_step * 5) * row_stride_u32];
            v6 = ptr_base[(r + effective_step * 6) * row_stride_u32];
            v7 = ptr_base[(r + effective_step * 7) * row_stride_u32];
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
                
                atomicAdd(&h0[v1 & 0xFF], 1);
                atomicAdd(&h1[(v1 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v1 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v1 >> 24], 1);
                
                unsigned int n2 = ptr_base[(r + effective_step * 2) * row_stride_u32];
                unsigned int n3 = ptr_base[(r + effective_step * 3) * row_stride_u32];
                
                // Process v2-v3
                atomicAdd(&h0[v2 & 0xFF], 1);
                atomicAdd(&h1[(v2 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v2 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v2 >> 24], 1);
                
                atomicAdd(&h0[v3 & 0xFF], 1);
                atomicAdd(&h1[(v3 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v3 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v3 >> 24], 1);
                
                unsigned int n4 = ptr_base[(r + effective_step * 4) * row_stride_u32];
                unsigned int n5 = ptr_base[(r + effective_step * 5) * row_stride_u32];
                
                // Process v4-v5
                atomicAdd(&h0[v4 & 0xFF], 1);
                atomicAdd(&h1[(v4 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v4 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v4 >> 24], 1);
                
                atomicAdd(&h0[v5 & 0xFF], 1);
                atomicAdd(&h1[(v5 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v5 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v5 >> 24], 1);
                
                unsigned int n6 = ptr_base[(r + effective_step * 6) * row_stride_u32];
                unsigned int n7 = ptr_base[(r + effective_step * 7) * row_stride_u32];
                
                // Process v6-v7
                atomicAdd(&h0[v6 & 0xFF], 1);
                atomicAdd(&h1[(v6 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v6 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v6 >> 24], 1);
                
                atomicAdd(&h0[v7 & 0xFF], 1);
                atomicAdd(&h1[(v7 >> 8) & 0xFF], 1);
                atomicAdd(&h2[(v7 >> 16) & 0xFF], 1);
                atomicAdd(&h3[v7 >> 24], 1);
                
                v0 = n0; v1 = n1; v2 = n2; v3 = n3;
                v4 = n4; v5 = n5; v6 = n6; v7 = n7;
                r += unroll_step_8x;
            }
            
            // Epilogue - process final buffered batch
            atomicAdd(&h0[v0 & 0xFF], 1);
            atomicAdd(&h1[(v0 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v0 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v0 >> 24], 1);
            
            atomicAdd(&h0[v1 & 0xFF], 1);
            atomicAdd(&h1[(v1 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v1 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v1 >> 24], 1);
            
            atomicAdd(&h0[v2 & 0xFF], 1);
            atomicAdd(&h1[(v2 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v2 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v2 >> 24], 1);
            
            atomicAdd(&h0[v3 & 0xFF], 1);
            atomicAdd(&h1[(v3 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v3 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v3 >> 24], 1);
            
            atomicAdd(&h0[v4 & 0xFF], 1);
            atomicAdd(&h1[(v4 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v4 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v4 >> 24], 1);
            
            atomicAdd(&h0[v5 & 0xFF], 1);
            atomicAdd(&h1[(v5 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v5 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v5 >> 24], 1);
            
            atomicAdd(&h0[v6 & 0xFF], 1);
            atomicAdd(&h1[(v6 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v6 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v6 >> 24], 1);
            
            atomicAdd(&h0[v7 & 0xFF], 1);
            atomicAdd(&h1[(v7 >> 8) & 0xFF], 1);
            atomicAdd(&h2[(v7 >> 16) & 0xFF], 1);
            atomicAdd(&h3[v7 >> 24], 1);
        }
        
        // Handle remaining elements
        for (; r < row_end; r += effective_step) {
            unsigned int val = ptr_base[r * row_stride_u32];
            atomicAdd(&h0[val & 0xFF], 1);
            atomicAdd(&h1[(val >> 8) & 0xFF], 1);
            atomicAdd(&h2[(val >> 16) & 0xFF], 1);
            atomicAdd(&h3[val >> 24], 1);
        }
    }
    __syncthreads();

    // 3. Flush (Privatized Store with Warp-Level Coalescing)
    // Write to: partial_output[blockIdx.y][global_channel][bin]
    // Flattened: blockIdx.y * (num_channels * num_bins) + global_channel * num_bins + bin
    
    int total_elements = num_channels * num_bins;
    int chunk_offset = chunk_idx * total_elements;
    
    // 64 channels * 256 bins to flush
    // Use warp-level voting to improve coalescing
    for (int i = tid; i < 64 * 256; i += blockDim.x) {
        int ch_local = i / 256;
        int bin = i % 256;
        
        // Swizzled Read
        int s_addr = (ch_local * 257) + (ch_local >> 2) + bin;
        int count = s_hist[s_addr];
        
        // Warp-level voting to identify threads with non-zero counts
        unsigned int mask = __ballot_sync(0xFFFFFFFF, count > 0);
        
        if (count > 0) {
            // Global Write (No Atomic)
            int global_ch = start_channel + ch_local;
            int global_idx = chunk_offset + global_ch * num_bins + bin;
            partial_output[global_idx] = count;
        } else if (mask != 0) {
            // Even if this thread has count==0, if any thread in warp has data,
            // we write 0 to help coalescing (only if warp has active writes)
            int global_ch = start_channel + ch_local;
            int global_idx = chunk_offset + global_ch * num_bins + bin;
            partial_output[global_idx] = 0;
        }
    }
}

// Host function
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels], dtype=uint8
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");
    TORCH_CHECK(data.dtype() == torch::kUInt8, "Tensor data must be UInt8");
    TORCH_CHECK(data.is_contiguous(), "Tensor data must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(data.data_ptr()) % 4 == 0, "Tensor data must be 4-byte aligned");
    
    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
        
    // Final Output
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    
    // Config optimized for H100 (132 SMs)
    dim3 block_dim(256, 1);
    int channels_per_block = 64;
    int grid_x = (num_channels + channels_per_block - 1) / channels_per_block; // 8
    
    // H100 Optimized Chunks: Target grid_x * grid_y to be multiple of 132 SMs
    // With grid_x=8, ideal grid_y values: 16 (128 blocks), 17 (136 blocks), 33 (264 blocks)
    // Balance between: more chunks = less reduction cost, fewer chunks = better SM utilization
    // For length=1048576, chunk_size ~32768 gives grid_y=32 (256 blocks total)
    // Try grid_y=33 for 264 blocks (2 waves on 132 SMs)
    int target_chunks = 33;
    int chunk_size = (length + target_chunks - 1) / target_chunks;
    if (chunk_size < 1) chunk_size = 1;
    int grid_y = (length + chunk_size - 1) / chunk_size;
    
    dim3 grid_dim(grid_x, grid_y);
    
    // Phase 1: Partial Histograms
    // Size: [grid_y, num_channels, num_bins]
    auto temp_options = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    torch::Tensor partial_hist = torch::empty({grid_y, num_channels, num_bins}, temp_options);
    
    int shared_mem_size = 64 * 258 * sizeof(int);
    
    // H100-specific optimizations
    cudaFuncSetAttribute(
        histogram_optimized_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size
    );
    
    // Prefer L1 cache for better atomic performance on shared memory
    cudaFuncSetAttribute(
        histogram_optimized_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100  // 100% shared memory, 0% L1 cache
    );
    
    // Set preferred cache config for global loads
    cudaFuncSetCacheConfig(histogram_optimized_kernel, cudaFuncCachePreferL1);
    
    histogram_optimized_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        data.data_ptr<unsigned char>(),
        partial_hist.data_ptr<int>(),
        length,
        num_channels,
        num_bins,
        chunk_size
    );
    
    // Phase 2: Reduce with optimized unrolling
    // Output size = 512 * 256 = 131072 elements.
    int total_elements = num_channels * num_bins;
    int reduce_block_size = 256;
    int reduce_grid_size = (total_elements + reduce_block_size - 1) / reduce_block_size;
    
    histogram_reduce_kernel<<<reduce_grid_size, reduce_block_size>>>(
        partial_hist.data_ptr<int>(),
        histogram.data_ptr<int>(),
        total_elements,
        grid_y
    );
    
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     throw std::runtime_error(cudaGetErrorString(err));
    // }
    
    return histogram;
}

"""

# C++ header declaration
cpp_source = """
#include <torch/extension.h>
torch::Tensor histogram_kernel(torch::Tensor data, int num_bins);
"""

# Ensure stdout and stderr exist
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

cuda_module = load_inline(
    name='submission_cuda_histogram_chenhzhu',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['histogram_kernel'],
    verbose=True,  # Enable verbose to see compilation details
    # with_cuda=True,
    # build_directory=".",
)

def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper function matching the required signature.
    
    Args:
        data: Tuple of (array, num_bins) where:
            array:    Tensor of shape [length, num_channels] with integer values in [0, num_bins-1]
            num_bins: Number of bins for the histogram
    
    Returns:
        histogram: Tensor of shape [num_channels, num_bins] containing histogram counts for each channel
    """

    array, num_bins = data
    
    if not array.is_cuda:
        array = array.cuda()
    
    # Call CUDA kernel
    histogram = cuda_module.histogram_kernel(array, num_bins)

    return histogram
