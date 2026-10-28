# python/dlcm/ops/fused_ops.py
import torch
import torch.nn.functional as F
from typing import Tuple

try:
    # These now return native numpy.ndarray objects directly!
    from dlcm._rust_ops import fused_boundary_op_py as rust_fused_boundary_op
    from dlcm._rust_ops import concept_replication_py as rust_concept_replication_op
    USE_RUST_KERNELS = True
    print("✅ [fused_ops] Rust kernels loaded. Using hardware-accelerated ops.")
except ImportError:
    USE_RUST_KERNELS = False
    print("⚠️ [fused_ops] Rust kernels not found. Falling back to pure PyTorch.")


def dynamic_segmentation_and_pooling(
    H: torch.Tensor, 
    W_q: torch.nn.Linear, 
    W_k: torch.nn.Linear, 
    W_up: torch.nn.Linear,
    alpha: float, 
    threshold: float, 
    training: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    if USE_RUST_KERNELS:
        # 1. Move to CPU and convert to numpy for the Rust bridge
        # (Note: For maximum performance, future iterations could use DLPack for zero-copy GPU transfer)
        h_np = H.cpu().numpy()
        wq_np = W_q.weight.data.cpu().numpy()
        wk_np = W_k.weight.data.cpu().numpy()
        wup_np = W_up.weight.data.cpu().numpy()
        
        # 2. Call Rust kernel (returns numpy arrays)
        c_np, b_np, p_np = rust_fused_boundary_op(h_np, wq_np, wk_np, wup_np, alpha, threshold, training)
        
        # 3. Convert back to PyTorch tensors on the original device
        device = H.device
        c = torch.from_numpy(c_np).to(device)
        b = torch.from_numpy(b_np).to(device)
        p = torch.from_numpy(p_np).to(device)
        
        return c, b, p
    
    # --- PURE PYTORCH FALLBACK (Unchanged) ---
    B, L, D = H.shape
    Q = W_q(H)
    K = W_k(H)
    Q_shifted = torch.zeros_like(Q)
    Q_shifted[:, 1:, :] = Q[:, :-1, :]
    
    cos_sim = F.cosine_similarity(Q_shifted, K, dim=-1)
    p = (1.0 - cos_sim) / 2.0
    p[:, 0] = 1.0
    
    if training:
        eps = 1e-6
        logits = torch.log(torch.clamp(p, eps, 1-eps) / torch.clamp(1-p, eps, 1-eps)) / alpha
        p_sharp = torch.sigmoid(logits)
        b = torch.bernoulli(p_sharp)
    else:
        b = (p >= threshold).float()
        
    segment_ids = torch.cumsum(b, dim=1) - 1
    max_M = int(segment_ids[:, -1].max().item()) + 1
    
    C_raw = torch.zeros(B, max_M, D, device=H.device, dtype=H.dtype)
    counts = torch.zeros(B, max_M, 1, device=H.device, dtype=H.dtype)
    
    idx = segment_ids.unsqueeze(-1).expand(-1, -1, D)
    C_raw.scatter_add_(1, idx, H)
    
    idx_counts = segment_ids.unsqueeze(-1)
    counts.scatter_add_(1, idx_counts, torch.ones(B, L, 1, device=H.device, dtype=H.dtype))
    
    C_raw = C_raw / (counts + 1e-8)
    C = W_up(C_raw)
    
    return C, b, p


def concept_replication(
    K: torch.Tensor, 
    V: torch.Tensor, 
    segment_lengths: torch.Tensor,
    target_L: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    if USE_RUST_KERNELS:
        k_np = K.cpu().numpy()
        v_np = V.cpu().numpy()
        seg_lens_np = segment_lengths.cpu().numpy()
        
        k_rep_np, v_rep_np = rust_concept_replication_op(k_np, v_np, seg_lens_np, target_L)
        
        device = K.device
        k_rep = torch.from_numpy(k_rep_np).to(device)
        v_rep = torch.from_numpy(v_rep_np).to(device)
        
        return k_rep, v_rep

    # --- PURE PYTORCH FALLBACK (Unchanged) ---
    B, M, num_heads, d_head = K.shape
    device = K.device
    
    K_rep_list = []
    V_rep_list = []
    
    for i in range(B):
        lengths = segment_lengths[i].cpu().tolist()
        current_sum = sum(lengths)
        if current_sum > target_L:
            lengths[-1] -= (current_sum - target_L)
        elif current_sum < target_L:
            lengths[-1] += (target_L - current_sum)
            
        k_rep = torch.repeat_interleave(K[i], torch.tensor(lengths, device=device), dim=0)
        v_rep = torch.repeat_interleave(V[i], torch.tensor(lengths, device=device), dim=0)
        
        K_rep_list.append(k_rep)
        V_rep_list.append(v_rep)
        
    return torch.stack(K_rep_list), torch.stack(V_rep_list)
