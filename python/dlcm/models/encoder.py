import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Eq. 16 context)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class EncoderAttention(nn.Module):
    """Standard Causal Self-Attention for the Token Encoder."""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_head, d_model, bias=False)
        
        # QK Normalization (Section 4)
        self.rms_norm_q = RMSNorm(self.d_head)
        self.rms_norm_k = RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        
        # Apply QK Normalization to stabilize training (Eq. 16)
        Q = self.rms_norm_q(Q)
        K = self.rms_norm_k(K)
        
        # FlashAttention / SDPA with causal mask
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        
        return self.W_o(attn_out)

class EncoderMLP(nn.Module):
    """SwiGLU Feed-Forward Network (Table 3)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish/SiLU activation as specified in Table 3
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class EncoderLayer(nn.Module):
    """Pre-norm Transformer Encoder Layer."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = EncoderAttention(d_model, num_heads)
        self.mlp = EncoderMLP(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TokenEncoder(nn.Module):
    """
    Stage 1: Encoding (Section 3.2)
    Processes raw tokens to extract fine-grained representations H.
    """
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
