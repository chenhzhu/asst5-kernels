import torch
import tilelang
import tilelang.language as T

# Test: Use Fragment for scalar accumulator
@tilelang.jit(out_idx=[1], target="cuda")
def test_sum_row_fixed(M, N):
    """Sum each row using Fragment for accumulator"""
    @T.prim_func
    def main(
        Input: T.Tensor([M, N], "float16"),
        Output: T.Tensor([M], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            # Use Fragment instead of local scalar
            acc = T.alloc_fragment([1], "float32")
            T.fill(acc, 0)
            
            for j in T.serial(N):
                acc[0] += T.float32(Input[i, j])
            
            Output[i] = T.cast(acc[0], "float16")
    return main

if __name__ == "__main__":
    M, N = 32, 10
    
    print("Test: Row Sum with Fragment Accumulator")
    a = torch.ones(M, N, device='cuda', dtype=torch.float16)
    kernel = test_sum_row_fixed(M, N)
    result = kernel(a)
    
    print(f"Input: all ones, shape {a.shape}")
    print(f"Result sample: {result[:5]}")
    print(f"Expected: all {N}")
    
    if torch.allclose(result, torch.full_like(result, N), atol=0.1):
        print("✅ Fragment accumulation works!")
    else:
        print(f"❌ Failed! Max: {result.max().item()}, Min: {result.min().item()}")

