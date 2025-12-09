#!/usr/bin/env python3
"""
Manually benchmark different CrossAttention configurations to find the best one.
"""
import torch
import time
import torch.nn.functional as F
from decoder_submission import cross_attention

def benchmark_config(config, q_proj, k_proj, v_proj, num_runs=20):
    """Benchmark a single configuration"""
    batch_size, num_heads, num_queries, head_dim = q_proj.shape
    _, _, num_latents, _ = k_proj.shape
    
    try:
        # Compile kernel
        kernel = cross_attention(
            batch_size,
            num_queries,
            num_latents,
            head_dim,
            **config
        )
        
        # Warmup
        for _ in range(5):
            _ = kernel(q_proj, k_proj, v_proj)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = kernel(q_proj, k_proj, v_proj)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times) * 1000  # Convert to ms
        return avg_time, output
        
    except Exception as e:
        print(f"  ❌ Config failed: {e}")
        return float('inf'), None

def main():
    print("=" * 80)
    print("CrossAttention Configuration Benchmark on H100")
    print("=" * 80)
    
    # Test workload (matching the assignment)
    batch_size = 1
    num_queries = 250000
    num_latents = 1024
    width = 768
    num_heads = 12
    head_dim = width // num_heads
    
    # Create dummy inputs
    print(f"\nWorkload: B={batch_size}, Nq={num_queries}, Nl={num_latents}, heads={num_heads}, head_dim={head_dim}")
    
    q_proj = torch.randn(batch_size, num_heads, num_queries, head_dim, device='cuda', dtype=torch.float16)
    k_proj = torch.randn(batch_size, num_heads, num_latents, head_dim, device='cuda', dtype=torch.float16)
    v_proj = torch.randn(batch_size, num_heads, num_latents, head_dim, device='cuda', dtype=torch.float16)
    
    # Configurations to test
    configs = [
        {"block_M": 128, "block_N": 128, "num_stages": 1, "threads": 256, "name": "M128_N128_S1"},
        {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256, "name": "M128_N128_S2"},
        {"block_M": 256, "block_N": 128, "num_stages": 1, "threads": 256, "name": "M256_N128_S1"},
        {"block_M": 128, "block_N": 64, "num_stages": 1, "threads": 256, "name": "M128_N64_S1"},
        {"block_M": 128, "block_N": 64, "num_stages": 2, "threads": 256, "name": "M128_N64_S2"},
        {"block_M": 64, "block_N": 128, "num_stages": 2, "threads": 256, "name": "M64_N128_S2"},
    ]
    
    print("\nTesting configurations...")
    print("-" * 80)
    
    results = []
    for config in configs:
        name = config.pop("name")
        print(f"\n📊 Testing: {name}")
        print(f"   {config}")
        
        avg_time, output = benchmark_config(config, q_proj, k_proj, v_proj)
        
        if output is not None:
            print(f"   ⏱️  {avg_time:.2f} ms")
            results.append((name, config, avg_time))
        
        config["name"] = name  # Put it back
    
    # Sort by time
    results.sort(key=lambda x: x[2])
    
    print("\n" + "=" * 80)
    print("RESULTS (sorted by speed)")
    print("=" * 80)
    
    for i, (name, config, avg_time) in enumerate(results):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{emoji} {name:20s}: {avg_time:6.2f} ms - {config}")
    
    if results:
        print("\n" + "=" * 80)
        print(f"✨ WINNER: {results[0][0]} with {results[0][2]:.2f} ms")
        print(f"   Config: {results[0][1]}")
        print("=" * 80)

if __name__ == "__main__":
    main()

