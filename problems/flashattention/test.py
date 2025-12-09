import torch
import triton
import triton.language as tl
import math
from task import input_t, output_t

def get_configs():
    configs = [
        # Hopper / H100-oriented config set.
        # We keep tiles <= 128x128 to avoid excessive register pressure while
        # giving autotune enough variety to choose from for different (seq_len, head_dim).
        #
        # Smaller tiles + fewer warps: good for short sequences / small heads.
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        # Rectangular vs square tiles for longer sequences.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
    ]
    return configs

@triton.autotune(
    configs=get_configs(),
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    sm_scale,
    seq_len, head_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Offsets
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    # -----------------------------------------------------------
    # L2 Cache Swizzling (Block Reordering)
    # -----------------------------------------------------------
    # We group blocks of Q to reuse K/V data in L2 cache.
    # Ideally, if we have 8 SMs, we want 8 Q-blocks to run in parallel
    # all accessing the same K/V blocks.
    
    num_pid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    
    # GROUP_SIZE_M = 8 is a common heuristic
    GROUP_SIZE_M = 8  

    # Calculate reordered pid_m
    group_id = pid_m // GROUP_SIZE_M
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Swizzle: execute row-major within the group
    # This maps the linear program_id to a swizzled ID
    pid_m = first_pid_m + (pid_m % group_size_m)
    # -----------------------------------------------------------

    # Memory offsets
    # Q: (B, H, S, D)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    
    # Pointers to Q
    # q_ptrs calculation
    # Base: Q + b*stride_b + h*stride_h
    # Offset: m*stride_m + d*stride_d
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    
    # Load Q
    # Mask out of bounds
    mask_m = offs_m < seq_len
    # Mask D if head_dim < BLOCK_SIZE_D
    mask_d = offs_d < head_dim
    
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # Initialize accumulators
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) # Initialized to 0
    acc_o = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D], dtype=tl.float32)
    
    # Pointers to K and V
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    
    # Loop over K, V blocks
    # Iterate over N dimension
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        # Pointers
        # Load K as (BLOCK_SIZE_N, BLOCK_SIZE_D)
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        mask_n = offs_n < seq_len
        
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Compute pairwise scores
        # q: (M, D), k: (N, D) -> q @ k.T : (M, N)
        # We use tl.dot
        qk = tl.dot(q, tl.trans(k))
        
        qk *= sm_scale
        
        # Mask out-of-bounds keys
        # We only need to mask if we are at the boundary or if causal masking (not here)
        if start_n + BLOCK_SIZE_N > seq_len:
             mask_n_row = offs_n[None, :] < seq_len
             qk = tl.where(mask_n_row, qk, float('-inf'))
        
        # --- Online Softmax ---
        
        # 1. Current block max
        m_ij = tl.max(qk, 1)
        
        # 2. Update global max
        m_new = tl.maximum(m_i, m_ij)
        
        # 3. Compute correction factors
        # alpha = exp(m_i - m_new)
        # beta = exp(m_ij - m_new)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        # 4. Update l_i
        # p_ij_unscaled = exp(qk - m_ij)
        # l_ij = sum(p_ij_unscaled)
        # l_new = alpha * l_i + beta * l_ij
        
        p = tl.exp(qk - m_new[:, None]) # Equivalent to exp(qk - m_ij) * beta
        
        # Check for potential optimization:
        # If we compute p using m_new, then sum(p) is l_local * beta
        # So l_new = l_i * alpha + sum(p)
        
        l_new = l_i * alpha + tl.sum(p, 1)
        
        # 5. Update acc_o
        # acc_o = acc_o * alpha + p @ v
        
        # p is (M, N), v is (N, D) -> (M, D)
        # p is fp32. v is fp16.
        # tl.dot supports mixed precision.
        
        acc_o = acc_o * alpha[:, None]
        acc_o += tl.dot(p.to(v.dtype), v)
        
        # Update state
        m_i = m_new
        l_i = l_new

    # Final normalization
    # o = acc_o / l_i
    o = acc_o / l_i[:, None]
    
    # Store O
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    
    tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are on CUDA and in FP16
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    
    # Extract dimensions
    batch_size, num_heads, seq_len, head_dim = Q.shape
    sm_scale = 1.0 / (head_dim ** 0.5)
    
    # Create output tensor
    O = torch.empty_like(Q)
    
    # Block size for Head Dim
    # Choose BLOCK_SIZE_D based on head_dim. We keep BLOCK_SIZE_D >= head_dim
    # so we do not need to tile over D inside the kernel.
    # Common transformer head sizes on H100 are 64 and 128.
    if head_dim <= 32:
        BLOCK_SIZE_D = 32
    elif head_dim <= 64:
        BLOCK_SIZE_D = 64
    elif head_dim <= 128:
        BLOCK_SIZE_D = 128
    else:
        # For larger head dimensions, fall back to the next power of 2.
        BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    grid = lambda META: (
        batch_size, 
        num_heads, 
        triton.cdiv(seq_len, META['BLOCK_SIZE_M'])
    )
    
    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sm_scale,
        seq_len, head_dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    # Print best config after autotuning
    # print("Best config:", flash_attention_kernel.best_config)
    
    return O

def custom_kernel(data: input_t) -> output_t:
    q, k, v = data
    return flash_attention_forward(q, k, v)
