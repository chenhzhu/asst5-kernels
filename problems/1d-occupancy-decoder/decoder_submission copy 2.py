import math
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from task import input_t, output_t

# Constants
dtype = "float16"
accum_dtype = "float"
num_heads = 12
eps = 1e-6

# ============================================================================
# MLPEmbedder: queries [B, N_q, 3] -> [B, N_q, width]
# ============================================================================

@tilelang.jit(
    out_idx=[5],  # Output is the 6th parameter (index 5)
    target="cuda",  # Explicitly set target to avoid auto-detection failure on WSL
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def mlp_embedder(
    batch,
    num_queries,
    block_M=32,  # Reduced from 128 to avoid Shared Memory overflow (384KB -> 96KB)
    threads=256
):
    # Width and input dim are fixed by the problem spec
    q_in_dim = 3
    width = 768
    block_N = 128  # tile the output width to keep fragments small
    input_shape = [batch, num_queries, q_in_dim]
    output_shape = [batch, num_queries, width]
    
    @T.prim_func
    def main(
        Queries: T.Tensor(input_shape, dtype),
        InWeight: T.Tensor([width, q_in_dim], dtype),
        InBias: T.Tensor([width], dtype),
        OutWeight: T.Tensor([width, width], dtype),
        OutBias: T.Tensor([width], dtype),
        Output: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(num_queries, block_M), batch, threads=threads) as (bx, bz):
            # Shared memory buffers
            Q_shared = T.alloc_shared([block_M, q_in_dim], dtype)
            Hidden_shared = T.alloc_shared([block_M, width], dtype)
            Output_shared = T.alloc_shared([block_M, width], dtype)
            
            # Load input tile - use element-wise copy to avoid TMA alignment requirement
            for i, j in T.Parallel(block_M, q_in_dim):
                q_idx = bx * block_M + i
                Q_shared[i, j] = T.if_then_else(
                    q_idx < num_queries,
                    Queries[bz, q_idx, j],
                    0
                )
            
            # First linear layer: Q @ InWeight^T + InBias
            # Q_shared: [block_M, 3], InWeight: [width, 3]
            # Manually compute GEMM because K=3 is too small/unaligned for Tensor Core GEMM
            # Directly apply Bias and SiLU here to save memory access
            for i, j in T.Parallel(block_M, width):
                val = T.float32(0)
                for k in T.serial(q_in_dim):
                    val += T.float32(Q_shared[i, k]) * T.float32(InWeight[j, k])
                
                x = val + T.float32(InBias[j])
                # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                sigmoid_x = 1.0 / (1.0 + T.exp(-x))
                Hidden_shared[i, j] = T.cast(x * sigmoid_x, dtype)
            
            # Second linear layer: Hidden @ OutWeight^T + OutBias
            # Use manual loop, write to shared first
            for i, j in T.Parallel(block_M, width):
                val = T.float32(0)
                for k in T.serial(width):
                    val += T.float32(Hidden_shared[i, k]) * T.float32(OutWeight[j, k])
                Output_shared[i, j] = T.cast(val + T.float32(OutBias[j]), dtype)
            
            # Write output with element-wise copy
            for i, j in T.Parallel(block_M, width):
                q_idx = bx * block_M + i
                # Write only if valid (no OOB write in TileLang auto-handles this)
                Output[bz, q_idx, j] = Output_shared[i, j]
    
    return main


# ============================================================================
# CrossAttention: FlashAttention style
# Input: Q [B, heads, N_q, head_dim], K [B, heads, N_l, head_dim], V [B, heads, N_l, head_dim]
# Output: [B, heads, N_q, head_dim]
# ============================================================================

@tilelang.jit(
    out_idx=[3],
    target="cuda",  # Explicitly set target
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def cross_attention(
    batch,
    num_queries,
    num_latents,
    head_dim,
    block_M=128,
    block_N=128,
    num_stages=2,
    threads=256
):
    q_shape = [batch, num_heads, num_queries, head_dim]
    kv_shape = [batch, num_heads, num_latents, head_dim]
    
    scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e) for exp2
    
    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, head_dim], dtype),
        K_shared: T.SharedBuffer([block_N, head_dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        # Fill -inf for OOB positions
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.if_then_else(
                k * block_N + j >= num_latents,
                -T.infinity(accum_dtype),
                0
            )
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
    
    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, head_dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, head_dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
    
    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)
    
    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([block_M, head_dim], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, head_dim):
            acc_o[i, j] *= scores_scale[i]
    
    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(num_queries, block_M), num_heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, head_dim], dtype)
            K_shared = T.alloc_shared([block_N, head_dim], dtype)
            V_shared = T.alloc_shared([block_N, head_dim], dtype)
            O_shared = T.alloc_shared([block_M, head_dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            
            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            
            loop_range = T.ceildiv(num_latents, block_N)
            
            for k in T.Pipelined(
                loop_range,
                num_stages=num_stages,
                order=[-1, 0, 3, 1, -1, 2],
                stage=[-1, 0, 0, 1, -1, 1],
                group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]]
            ):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            
            for i, j in T.Parallel(block_M, head_dim):
                acc_o[i, j] /= logsum[i]
            
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])
    
    return main


# ============================================================================
# LayerNorm: [B, N_q, width] -> [B, N_q, width]
# ============================================================================

@tilelang.jit(
    out_idx=[1],  # Output is the 2nd parameter (index 1)
    target="cuda",  # Explicitly set target
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def layer_norm(
    batch,
    num_queries,
    block_M=32,  # Reduced from 128 to avoid Shared Memory overflow
    threads=256
):
    width = 768
    input_shape = [batch, num_queries, width]
    
    @T.prim_func
    def main(
        Input: T.Tensor(input_shape, dtype),
        Output: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(num_queries, block_M), batch, threads=threads) as (bx, bz):
            Input_shared = T.alloc_shared([block_M, width], dtype)
            Output_shared = T.alloc_shared([block_M, width], dtype)
            
            # Fragment buffers for mean and variance computation
            mean_acc = T.alloc_fragment([block_M], accum_dtype)
            var_acc = T.alloc_fragment([block_M], accum_dtype)
            mean_val = T.alloc_fragment([block_M], accum_dtype)
            var_val = T.alloc_fragment([block_M], accum_dtype)
            
            T.copy(Input[bz, bx * block_M:(bx + 1) * block_M, :], Input_shared)
            
            # Compute mean for each row
            # Use serial loop for reduction to avoid race condition in T.Parallel
            for i in T.Parallel(block_M):
                val = T.float32(0)
                for j in T.serial(width):
                    val += T.float32(Input_shared[i, j])
                mean_val[i] = val / width
            
            # Compute variance for each row
            for i in T.Parallel(block_M):
                val = T.float32(0)
                mean_i = mean_val[i]
                for j in T.serial(width):
                    diff = T.float32(Input_shared[i, j]) - mean_i
                    val += diff * diff
                
                var_val[i] = val / width + eps
                var_val[i] = 1.0 / T.sqrt(var_val[i])
            
            # Normalize
            for i, j in T.Parallel(block_M, width):
                Output_shared[i, j] = (Input_shared[i, j] - mean_val[i]) * var_val[i]
            
            T.copy(Output_shared, Output[bz, bx * block_M:(bx + 1) * block_M, :])
    
    return main


# ============================================================================
# Output Projection: [B, N_q, width] -> [B, N_q, 1]
# ============================================================================

@tilelang.jit(
    out_idx=[3],  # Output is the 4th parameter (index 3)
    target="cuda",  # Explicitly set target
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def output_projection(
    batch,
    num_queries,
    block_M=32,  # Reduced from 128 for consistency and safety
    threads=256
):
    width = 768
    out_features = 1
    input_shape = [batch, num_queries, width]
    output_shape = [batch, num_queries, out_features]
    
    @T.prim_func
    def main(
        Input: T.Tensor(input_shape, dtype),
        Weight: T.Tensor([out_features, width], dtype),
        Bias: T.Tensor([out_features], dtype),
        Output: T.Tensor(output_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(num_queries, block_M), batch, threads=threads) as (bx, bz):
            Input_shared = T.alloc_shared([block_M, width], dtype)
            Output_shared = T.alloc_shared([block_M, out_features], dtype)
            # acc removed
            
            T.copy(Input[bz, bx * block_M:(bx + 1) * block_M, :], Input_shared)
            # Manually compute GEMM because N=1 is too small/unaligned for Tensor Core GEMM
            for i, j in T.Parallel(block_M, out_features):
                val = T.float32(0)
                for k in T.serial(width):
                    val += T.float32(Input_shared[i, k]) * T.float32(Weight[j, k])
                # Add bias and cast to float16
                Output_shared[i, j] = T.cast(val + T.float32(Bias[j]), dtype)
            
            # Write output - use element-wise copy to avoid TMA alignment requirement
            # For small out_features (1), element-wise copy is efficient
            for i, j in T.Parallel(block_M, out_features):
                q_idx = bx * block_M + i
                # Only write if within valid range
                val = T.if_then_else(q_idx < num_queries, Output_shared[i, j], 0)
                Output[bz, q_idx, j] = val
    
    return main


# ============================================================================
# Main custom_kernel function
# ============================================================================

# Pre-allocate global output tensor to avoid allocation overhead in benchmark
_OUT_CACHE = None

def custom_kernel(data: input_t) -> output_t:
    """
    TileLang implementation of OneDOccupancyDecoder forward pass.
    
    Args:
        data: Tuple of (queries, latents, weights)
        
    Returns:
        Output tensor of shape [batch_size, num_queries, 1]
    """
    global _OUT_CACHE
    queries, latents, weights = data
    
    # Ensure float16
    if queries.dtype != torch.float16:
        queries = queries.half()
        latents = latents.half()
        for key in weights:
            weights[key] = weights[key].half()
    
    batch_size, num_queries, q_in_dim = queries.shape
    _, num_latents, width = latents.shape
    head_dim = width // num_heads
    
    # Pre-allocate output tensor to avoid allocation overhead
    if _OUT_CACHE is None or _OUT_CACHE.shape != (batch_size, num_queries, 1) or _OUT_CACHE.device != queries.device:
        _OUT_CACHE = torch.empty(
            (batch_size, num_queries, 1), dtype=torch.float16, device=queries.device
        )
    out = _OUT_CACHE
    
    # Step 1: MLPEmbedder (using PyTorch for small dimensions)
    # First layer: [B, N_q, 3] -> [B, N_q, 768]
    x = F.linear(queries, weights['query_in_in_layer_weight'], weights['query_in_in_layer_bias'])
    x = F.silu(x)  # SiLU activation
    embedded_queries = F.linear(x, weights['query_in_out_layer_weight'], weights['query_in_out_layer_bias'])
    
    # Step 2: CrossAttention
    # Project queries to Q
    q_proj = F.linear(
        embedded_queries,
        weights['attn_c_q_weight'],
        weights['attn_c_q_bias']
    )
    # Reshape: [B, N_q, width] -> [B, heads, N_q, head_dim]
    q_proj = q_proj.view(batch_size, num_queries, num_heads, head_dim).transpose(1, 2).contiguous()
    
    # Project latents to K and V
    k_proj = F.linear(
        latents,
        weights['attn_c_k_weight'],
        weights['attn_c_k_bias']
    )
    v_proj = F.linear(
        latents,
        weights['attn_c_v_weight'],
        weights['attn_c_v_bias']
    )
    # Reshape: [B, N_l, width] -> [B, heads, N_l, head_dim]
    k_proj = k_proj.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2).contiguous()
    v_proj = v_proj.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2).contiguous()
    
    # FlashAttention (tuned parameters for H100)
    # block_M=128, block_N=128 is stable, try reducing stages to lower register pressure
    attn_kernel = cross_attention(
        batch_size,
        num_queries,
        num_latents,
        head_dim,
        block_M=128,
        block_N=128,
        num_stages=1,  # Reduce from 2 to 1 to avoid register serialization
        threads=256
    )
    attn_output = attn_kernel(q_proj, k_proj, v_proj)
    
    # Reshape back: [B, heads, N_q, head_dim] -> [B, N_q, width]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, width)
    
    # Project attention output
    attn_output = F.linear(
        attn_output,
        weights['attn_c_proj_weight'],
        weights['attn_c_proj_bias']
    )
    
    # Step 3: LayerNorm (using PyTorch)
    # LayerNorm with elementwise_affine=False, eps=1e-6
    # Compute in float32 for numerical stability, then cast back to float16
    x_float = attn_output.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, unbiased=False)
    normalized = ((x_float - mean) / torch.sqrt(var + eps)).half()
    
    # Step 4: Output Projection (using PyTorch for small output dimension)
    # Use pre-allocated output buffer
    F.linear(normalized, weights['out_proj_weight'], weights['out_proj_bias'], out=out)
    
    return out
