import torch
import tilelang
import tilelang.language as T

# Test 1: Simple accumulation
@tilelang.jit(out_idx=[1], target="cuda")
def test_sum_row(M, N):
    """Sum each row"""
    @T.prim_func
    def main(
        Input: T.Tensor([M, N], "float16"),
        Output: T.Tensor([M], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            val = T.float32(0)
            for j in T.serial(N):
                val += T.float32(Input[i, j])
            Output[i] = T.cast(val, "float16")
    return main

# Test 2: Direct write without accumulation
@tilelang.jit(out_idx=[1], target="cuda")
def test_direct_write(M, N):
    """Just write first element of each row"""
    @T.prim_func
    def main(
        Input: T.Tensor([M, N], "float16"),
        Output: T.Tensor([M, N], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            for j in T.serial(N):
                Output[i, j] = Input[i, 0]  # Write same value to all columns
    return main

if __name__ == "__main__":
    M, N = 32, 10
    
    # Test 1: Sum
    print("Test 1: Row Sum with Accumulation")
    a = torch.ones(M, N, device='cuda', dtype=torch.float16)
    kernel1 = test_sum_row(M, N)
    result1 = kernel1(a)
    
    print(f"Input: all ones, shape {a.shape}")
    print(f"Result shape: {result1.shape}")
    print(f"Result sample: {result1[:5]}")
    print(f"Expected: all {N} (sum of {N} ones)")
    
    if torch.allclose(result1, torch.full_like(result1, N), atol=0.1):
        print("✅ Accumulation works!\n")
    else:
        print(f"❌ Accumulation failed!")
        if result1.abs().sum().item() == 0:
            print("⚠️ Result is all zeros!\n")
    
    # Test 2: Direct write
    print("Test 2: Direct Write (no accumulation)")
    a2 = torch.arange(M * N, device='cuda', dtype=torch.float16).reshape(M, N)
    kernel2 = test_direct_write(M, N)
    result2 = kernel2(a2)
    
    print(f"Result sample (first row): {result2[0, :5]}")
    print(f"Expected: all {a2[0, 0].item()}")
    
    if torch.allclose(result2[0], torch.full_like(result2[0], a2[0, 0]), atol=0.1):
        print("✅ Direct write works!")
    else:
        print(f"❌ Direct write failed!")

