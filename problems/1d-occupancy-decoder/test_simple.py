import torch
import tilelang
import tilelang.language as T

# Test 1: Simplest possible kernel - just copy input to output
@tilelang.jit(out_idx=[1], target="cuda")
def simple_copy(M, N):
    @T.prim_func
    def main(
        Input: T.Tensor([M, N], "float16"),
        Output: T.Tensor([M, N], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            for j in T.serial(N):
                Output[i, j] = Input[i, j]
    return main

# Test 2: Simple add operation
@tilelang.jit(out_idx=[2], target="cuda")
def simple_add(M, N):
    @T.prim_func
    def main(
        A: T.Tensor([M, N], "float16"),
        B: T.Tensor([M, N], "float16"),
        Output: T.Tensor([M, N], "float16"),
    ):
        with T.Kernel(M, threads=256) as i:
            for j in T.serial(N):
                Output[i, j] = A[i, j] + B[i, j]
    return main

if __name__ == "__main__":
    M, N = 32, 768
    
    # Test 1: Copy
    print("Test 1: Simple Copy")
    a = torch.randn(M, N, device='cuda', dtype=torch.float16)
    kernel1 = simple_copy(M, N)
    result1 = kernel1(a)
    
    if torch.allclose(a, result1, atol=1e-3):
        print("✅ Copy works!")
    else:
        print(f"❌ Copy failed! Max diff: {(a - result1).abs().max().item()}")
        print(f"Input sample: {a[0, :5]}")
        print(f"Output sample: {result1[0, :5]}")
    
    # Test 2: Add
    print("\nTest 2: Simple Add")
    b = torch.randn(M, N, device='cuda', dtype=torch.float16)
    kernel2 = simple_add(M, N)
    result2 = kernel2(a, b)
    expected = a + b
    
    if torch.allclose(expected, result2, atol=1e-3):
        print("✅ Add works!")
    else:
        print(f"❌ Add failed! Max diff: {(expected - result2).abs().max().item()}")
        print(f"Expected sample: {expected[0, :5]}")
        print(f"Output sample: {result2[0, :5]}")

