import torch
import torch.nn.functional as F
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from task import input_t, output_t, ModelWeights
from reference import ref_kernel, generate_input
from model import OneDOccupancyDecoder
from decoder_submission import cross_attention

def debug_compare():
    print("=== Debugging TileLang Implementation Step-by-Step ===")
    
    # 1. Setup Inputs (Small Scale)
    # Use a very small shape to make debugging easier and avoid potential OOM/timeout on local
    batch_size = 1
    num_queries = 32
    num_latents = 16
    width = 768 # Keep width same as it matches weight shapes
    num_heads = 12
    q_in_dim = 3
    seed = 42
    
    print(f"Config: B={batch_size}, Nq={num_queries}, Nl={num_latents}, W={width}, H={num_heads}")

    # Tunable block sizes for local GPU (e.g., RTX 3060 has 64KB shared memory per SM)
    blk_m_mlp = 16
    blk_m_ln = 16
    blk_m_out = 16
    blk_m_attn = 64
    blk_n_attn = 64
    
    # Generate inputs
    queries, latents, weights = generate_input(
        batch_size, num_queries, num_latents, width, num_heads, q_in_dim, seed
    )
    
    # Ensure float16
    queries = queries.half()
    latents = latents.half()
    for key in weights:
        weights[key] = weights[key].half()

    # Setup Reference Model
    ref_model = OneDOccupancyDecoder(
        q_in_dim=q_in_dim,
        width=width,
        num_heads=num_heads,
        out_features=1,
    ).to('cuda', dtype=torch.float16)
    
    # Load weights to ref model
    ref_model.query_in.in_layer.weight.data.copy_(weights['query_in_in_layer_weight'])
    ref_model.query_in.in_layer.bias.data.copy_(weights['query_in_in_layer_bias'])
    ref_model.query_in.out_layer.weight.data.copy_(weights['query_in_out_layer_weight'])
    ref_model.query_in.out_layer.bias.data.copy_(weights['query_in_out_layer_bias'])
    
    ref_model.attn.c_q.weight.data.copy_(weights['attn_c_q_weight'])
    ref_model.attn.c_q.bias.data.copy_(weights['attn_c_q_bias'])
    ref_model.attn.c_k.weight.data.copy_(weights['attn_c_k_weight'])
    ref_model.attn.c_k.bias.data.copy_(weights['attn_c_k_bias'])
    ref_model.attn.c_v.weight.data.copy_(weights['attn_c_v_weight'])
    ref_model.attn.c_v.bias.data.copy_(weights['attn_c_v_bias'])
    ref_model.attn.c_proj.weight.data.copy_(weights['attn_c_proj_weight'])
    ref_model.attn.c_proj.bias.data.copy_(weights['attn_c_proj_bias'])
    
    ref_model.out_proj.weight.data.copy_(weights['out_proj_weight'])
    ref_model.out_proj.bias.data.copy_(weights['out_proj_bias'])
    
    ref_model.eval()

    # === Step 1: MLPEmbedder ===
    print("\n--- Step 1: MLPEmbedder ---")
    # PyTorch Ref
    with torch.no_grad():
        ref_q = ref_model.query_in(queries)
    
    # PyTorch implementation (same as custom_kernel)
    x = F.linear(queries, weights['query_in_in_layer_weight'], weights['query_in_in_layer_bias'])
    x = F.silu(x)
    tl_q = F.linear(x, weights['query_in_out_layer_weight'], weights['query_in_out_layer_bias'])
    
    check_diff(ref_q, tl_q, "MLPEmbedder")
    
    # Use reference output for next step to isolate errors
    curr_input = ref_q 

    # === Step 2: Cross Attention (Projection + Attention) ===
    print("\n--- Step 2: Cross Attention ---")
    
    # PyTorch Ref
    # ref_model.attn(x, c) -> q,k,v proj -> attention -> out proj
    with torch.no_grad():
        ref_attn_out = ref_model.attn(curr_input, latents)
        
    # TileLang Logic (mimicking custom_kernel)
    head_dim = width // num_heads
    
    # 2.1 Projections (using PyTorch F.linear as in submission)
    tl_q_proj = F.linear(curr_input, weights['attn_c_q_weight'], weights['attn_c_q_bias'])
    tl_q_proj = tl_q_proj.view(batch_size, num_queries, num_heads, head_dim).transpose(1, 2).contiguous()
    
    tl_k_proj = F.linear(latents, weights['attn_c_k_weight'], weights['attn_c_k_bias'])
    tl_v_proj = F.linear(latents, weights['attn_c_v_weight'], weights['attn_c_v_bias'])
    
    tl_k_proj = tl_k_proj.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2).contiguous()
    tl_v_proj = tl_v_proj.view(batch_size, num_latents, num_heads, head_dim).transpose(1, 2).contiguous()

    # 2.2 Flash Attention Kernel
    try:
        attn_k = cross_attention(
            batch_size, num_queries, num_latents, head_dim, 
            block_M=blk_m_attn, block_N=blk_n_attn, num_stages=2, threads=256
        )
        tl_attn_val = attn_k(tl_q_proj, tl_k_proj, tl_v_proj)
    except Exception as e:
        print(f"⚠️ CrossAttention compilation failed (expected on local GPU): {type(e).__name__}")
        print("   Using PyTorch implementation for local debugging...")
        tl_attn_val = F.scaled_dot_product_attention(tl_q_proj, tl_k_proj, tl_v_proj)
    
    # 2.3 Reshape and Out Project
    tl_attn_val = tl_attn_val.transpose(1, 2).contiguous().view(batch_size, num_queries, width)
    tl_attn_out = F.linear(tl_attn_val, weights['attn_c_proj_weight'], weights['attn_c_proj_bias'])
    
    check_diff(ref_attn_out, tl_attn_out, "CrossAttention (Full Block)")
    
    curr_input = ref_attn_out

    # === Step 3: LayerNorm ===
    print("\n--- Step 3: LayerNorm ---")
    # PyTorch Ref
    with torch.no_grad():
        ref_ln_out = ref_model.ln(curr_input)
        
    # PyTorch implementation (same as custom_kernel)
    eps = 1e-6
    x_float = curr_input.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = x_float.var(dim=-1, keepdim=True, unbiased=False)
    tl_ln_out = ((x_float - mean) / torch.sqrt(var + eps)).half()
    
    check_diff(ref_ln_out, tl_ln_out, "LayerNorm")
    
    curr_input = ref_ln_out

    # === Step 4: Output Projection ===
    print("\n--- Step 4: Output Projection ---")
    # PyTorch Ref
    with torch.no_grad():
        ref_out = ref_model.out_proj(curr_input)
        
    # PyTorch implementation (same as custom_kernel)
    tl_out = F.linear(curr_input, weights['out_proj_weight'], weights['out_proj_bias'])
    
    check_diff(ref_out, tl_out, "Output Projection")


def check_diff(ref, val, name):
    if ref.shape != val.shape:
        print(f"❌ {name}: Shape Mismatch! Ref {ref.shape} vs Val {val.shape}")
        return
        
    diff = torch.abs(ref - val)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # For float16, larger tolerance might be needed
    if torch.allclose(ref, val, atol=1e-2, rtol=1e-2):
        print(f"✅ {name}: Match! Max Diff: {max_diff:.6f}")
    else:
        print(f"❌ {name}: Mismatch! Max Diff: {max_diff:.6f}, Mean Diff: {mean_diff:.6f}")
        # Print sample
        print(f"Ref sample: {ref.flatten()[:5]}")
        print(f"Val sample: {val.flatten()[:5]}")

if __name__ == "__main__":
    debug_compare()
