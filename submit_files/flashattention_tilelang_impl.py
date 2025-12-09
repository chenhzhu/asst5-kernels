import math
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import itertools
from task import input_t, output_t

def get_configs():
    iter_params = dict(block_M=[128], block_N=[128], num_stages=[2], threads=[256])
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]

@tilelang.jit(
    out_idx=[3], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(batch,
              heads,
              seq_q,
              seq_kv,
              dim,
              is_causal,
              block_M=128,
              block_N=128,
              num_stages=2,
              threads=256):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = "float16"
    accum_dtype = "float"

    past_len = seq_kv - seq_q
    # assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, dim], dtype),
        K_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        else:
            # We shall fill -inf for OOB positions
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
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
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N), T.ceildiv(
                        (bx + 1) * block_M +
                        past_len, block_N)) if is_causal else T.ceildiv(seq_kv, block_N))

            for k in T.Pipelined(
                    loop_range,
                    num_stages=num_stages,
                    order=[-1, 0, 3, 1, -1, 2],
                    stage=[-1, 0, 0, 1, -1, 1],
                    group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]]):
                MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum,
                        logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main

def custom_kernel(data: input_t) -> output_t:
    """
    Wrapper matching the assignment's expected interface.
    """
    q, k, v = data
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # JIT compile and run the kernel
    # We use a cached kernel if shapes match to avoid recompilation overhead
    # In a real scenario we might need to handle variable shapes better or use the autotuner more smartly
    
    # Note: The original template uses torch.float16. We ensure inputs are float16.
    if q.dtype != torch.float16:
        q = q.half()
        k = k.half()
        v = v.half()
    
    is_causal = False
        
    # Use the optimal configuration found during tuning
    kernel = flashattn(
        batch_size,
        num_heads,
        seq_len,
        seq_len,
        head_dim,
        is_causal,
        block_M=128,
        block_N=128,
        num_stages=2,
        threads=256
    )
    return kernel(q, k, v)

