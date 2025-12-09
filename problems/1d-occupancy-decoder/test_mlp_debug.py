import torch
import tilelang
import tilelang.language as T

# Test: Multi-input GEMM-like operation
@tilelang.jit(out_idx=[2], target="cuda")
def test_matmul_manual(M, K, N):
    """
    Manual matmul: A[M, K] @ B[K, N] = C[M, N]
    """
    @T.prim_func
    def main(
        A: T.Tensor([M, K], "float16"),
        B: T.Tensor([N, K], "float16"),  # Note: B is [N, K] for transpose
        Output: T.Tensor([M, N], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            for j in T.serial(N):
                val = T.float32(0)
                for k in T.serial(K):
                    val += T.float32(A[i, k]) * T.float32(B[j, k])
                Output[i, j] = T.cast(val, "float16")
    return main

if __name__ == "__main__":
    M, K, N = 32, 3, 768
    
    print(f"Test: Manual MatMul [{M}, {K}] @ [{K}, {N}]^T = [{M}, {N}]")
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(N, K, device='cuda', dtype=torch.float16)
    
    kernel = test_matmul_manual(M, K, N)
    result = kernel(A, B)
    
    # PyTorch reference: A @ B.T
    expected = A @ B.T
    
    print(f"Result shape: {result.shape}")
    print(f"Result sample: {result[0, :5]}")
    print(f"Expected sample: {expected[0, :5]}")
    
    if torch.allclose(expected, result, atol=1e-2, rtol=1e-2):
        print("✅ Manual MatMul works!")
    else:
        max_diff = (expected - result).abs().max().item()
        mean_diff = (expected - result).abs().mean().item()
        print(f"❌ Manual MatMul failed! Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        
        # Check if result is all zeros
        if result.abs().sum().item() == 0:
            print("⚠️ Result is all zeros!")

