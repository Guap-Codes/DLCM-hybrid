import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import RMSNorm, EncoderMLP

class ConceptSmoothing(nn.Module):
    """
    Stage 4.1: Concept Smoothing (Section 3.5.1)
    Lightweight module to integrate adjacent concepts and reduce 
    discretization artifacts from hard pooling.
    """
    def __init__(self, d_concept: int):
        super().__init__()
        self.norm = RMSNorm(d_concept)
        self.proj = nn.Linear(d_concept, d_concept, bias=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(Z))

class DecoderCrossAttention(nn.Module):
    """
    Stage 4.2: Causal Cross-Attention (Section 3.5.2 & 4.1)
    Implements QK normalization, Concept Replication, and Cross-Attention.
    """
    def __init__(self, d_token: int, d_concept: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_token // num_heads
        
        self.W_q = nn.Linear(d_token, num_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_concept, num_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_concept, num_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_head, d_token, bias=False)
        
        # QK Normalization (Eq. 16)
        self.rms_norm_q = RMSNorm(self.d_head)
        self.rms_norm_k = RMSNorm(self.d_head)

    def forward(self, H: torch.Tensor, Z_smooth: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Encoder hidden states [B, L, d_token]
            Z_smooth: Smoothed concept states [B, M, d_concept]
            segment_lengths: Number of tokens in each concept [B, M]
        """
        B, L, _ = H.shape
        M = Z_smooth.shape[1]
        
        # Eq 12, 13: Project to common head dimension
        Q = self.W_q(H).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(Z_smooth).view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(Z_smooth).view(B, M, self.num_heads, self.d_head).transpose(1, 2)
        
        # Eq 16: QK Normalization
        Q = self.rms_norm_q(Q)
        K = self.rms_norm_k(K)
        
        # Eq 17: Concept Replication (The core optimization trick)
        # Replicate K and V to match token length L, enabling standard LxL causal masks
        rep_idx = self._get_replication_indices(segment_lengths, L)
        rep_idx_exp = rep_idx.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.d_head)
        
        K_rep = torch.gather(K, 2, rep_idx_exp) # [B, nh, L, dh]
        V_rep = torch.gather(V, 2, rep_idx_exp) # [B, nh, L, dh]
        
        # Eq 14: Attention with standard causal mask
        # Because K/V are replicated, a standard LxL causal mask naturally enforces 
        # that token t only attends to concepts formed up to j(t).
        attn_out = F.scaled_dot_product_attention(Q, K_rep, V_rep, is_causal=True)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(attn_out)

    def _get_replication_indices(self, segment_lengths: torch.Tensor, L: int) -> torch.Tensor:
        """Generates indices for torch.gather to simulate repeat_interleave."""
        B, M = segment_lengths.shape
        indices = torch.zeros(B, L, dtype=torch.long, device=segment_lengths.device)
        for i in range(B):
            lengths = segment_lengths[i]
            # Safety clamp to ensure sum of lengths exactly matches L
            if lengths.sum() > L:
                lengths[-1] -= (lengths.sum() - L)
            elif lengths.sum() < L:
                lengths[-1] += (L - lengths.sum())
                
            indices[i] = torch.repeat_interleave(torch.arange(M, device=segment_lengths.device), lengths)
        return indices

class DecoderLayer(nn.Module):
    """Pre-norm Transformer Decoder Layer with Cross-Attention."""
    def __init__(self, d_token: int, d_concept: int, num_heads: int, d_ff: int):
        super().__init__()
        self.cross_attn = DecoderCrossAttention(d_token, d_concept, num_heads)
        self.mlp = EncoderMLP(d_token, d_ff)
        self.norm1 = RMSNorm(d_token)
        self.norm2 = RMSNorm(d_token)

    def forward(self, H: torch.Tensor, Z_smooth: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
        # Cross-Attention with residual connection (Eq. 14)
        H = H + self.cross_attn(self.norm1(H), Z_smooth, segment_lengths)
        # MLP with residual connection
        H = H + self.mlp(self.norm2(H))
        return H

class TokenDecoder(nn.Module):
    """
    Stage 4: Token-Level Decoding (Section 3.5)
    Reconstructs token-level predictions by attending to reasoned concepts.
    """
    def __init__(self, d_token: int, d_concept: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.smoothing = ConceptSmoothing(d_concept)
        self.layers = nn.ModuleList([
            DecoderLayer(d_token, d_concept, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_token)
        
    def forward(self, H: torch.Tensor, Z: torch.Tensor, segment_lengths: torch.Tensor) -> torch.Tensor:
        Z_smooth = self.smoothing(Z)
        for layer in self.layers:
            H = layer(H, Z_smooth, segment_lengths)
        return self.norm(H)
