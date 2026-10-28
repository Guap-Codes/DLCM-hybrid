import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .encoder import TokenEncoder
from .concept_model import ConceptReasoningBackbone
from .decoder import TokenDecoder

@dataclass
class DLCMConfig:
    """Configuration matching the 2.3B DLCM architecture (Table 3)."""
    vocab_size: int = 128815
    d_token: int = 1536
    d_concept: int = 3072
    d_scan: int = 1536
    
    # Encoder (10 layers)
    enc_layers: int = 10
    enc_heads: int = 24
    enc_ff: int = 4096
    
    # Concept Backbone (16 layers)
    concept_layers: int = 16
    concept_heads: int = 48
    concept_ff: int = 6144
    
    # Decoder (6 layers)
    dec_layers: int = 6
    dec_heads: int = 24
    dec_ff: int = 4096
    
    # Segmentation Hyperparameters
    alpha: float = 1.0      # Temperature for Bernoulli sharpening
    threshold: float = 0.5  # Hard threshold for inference
    
    # muP Scaling Factor (Eq 21)
    s_token: float = 1.0 

class DynamicSegmentation(nn.Module):
    """
    Stage 2: Dynamic Segmentation & Pooling (Section 3.3)
    Learns semantic boundaries and pools tokens into concepts.
    """
    def __init__(self, d_token: int, d_scan: int, d_concept: int, alpha: float, threshold: float):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        
        # Boundary Detection Projections (Eq. 5)
        self.W_q = nn.Linear(d_token, d_scan, bias=False)
        self.W_k = nn.Linear(d_token, d_scan, bias=False)
        
        # Concept Dimension Projection (Eq. 7)
        self.W_up = nn.Linear(d_token, d_concept, bias=False)

    def forward(self, H: torch.Tensor, training: bool = True):
        """
        Args:
            H: Encoder hidden states [B, L, d_token]
            training: Boolean flag to switch between Bernoulli sampling and hard thresholding
        Returns:
            C: Pooled concept representations [B, max_M, d_concept]
            b: Boundary indicators [B, L]
            p: Boundary probabilities [B, L]
        """
        B, L, D = H.shape
        
        # Eq 5: Project to query-key space
        Q = self.W_q(H)
        K = self.W_k(H)
        
        # Eq 6: Compute boundary probability p_t = (1 - cos(q_{t-1}, k_t)) / 2
        Q_shifted = torch.zeros_like(Q)
        Q_shifted[:, 1:, :] = Q[:, :-1, :]
        
        cos_sim = F.cosine_similarity(Q_shifted, K, dim=-1)
        p = (1.0 - cos_sim) / 2.0
        
        # Enforce p_1 = 1 (First token always starts a new concept)
        p[:, 0] = 1.0
        
        # Discrete Sampling (Section 3.3.1)
        if training:
            # Sharpen probabilities by temperature alpha
            eps = 1e-6
            logits = torch.log(torch.clamp(p, eps, 1-eps) / torch.clamp(1-p, eps, 1-eps)) / self.alpha
            p_sharp = torch.sigmoid(logits)
            b = torch.bernoulli(p_sharp)
        else:
            b = (p >= self.threshold).float()
            
        # Concept Formation via Pooling (Eq. 7)
        # segment_ids = torch.cumsum(b, dim=1) - 1  # 0-indexed segment IDs [B, L]
        # max_M = int(segment_ids[:, -1].max().item()) + 1
        
        # Concept Formation via Pooling (Eq. 7)
        # FIX: Add .long() to cast the float cumsum output to int64
        segment_ids = (torch.cumsum(b, dim=1) - 1).long()  # 0-indexed segment IDs [B, L]
        max_M = int(segment_ids[:, -1].max().item()) + 1

        # Vectorized Mean Pooling using scatter_add_
        C_raw = torch.zeros(B, max_M, D, device=H.device, dtype=H.dtype)
        counts = torch.zeros(B, max_M, 1, device=H.device, dtype=H.dtype)
        
        idx = segment_ids.unsqueeze(-1).expand(-1, -1, D)
        C_raw.scatter_add_(1, idx, H)
        
        idx_counts = segment_ids.unsqueeze(-1)
        counts.scatter_add_(1, idx_counts, torch.ones(B, L, 1, device=H.device, dtype=H.dtype))
        
        C_raw = C_raw / (counts + 1e-8)
        C = self.W_up(C_raw)  # Project to d_concept
        
        return C, b, p

class DLCMModel(nn.Module):
    """
    Full Dynamic Large Concept Model Pipeline (Section 3.1)
    """
    def __init__(self, config: DLCMConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_token)
        self.encoder = TokenEncoder(config.d_token, config.enc_layers, config.enc_heads, config.enc_ff)
        self.segmentation = DynamicSegmentation(config.d_token, config.d_scan, config.d_concept, config.alpha, config.threshold)
        self.concept_model = ConceptReasoningBackbone(config.d_concept, config.concept_layers, config.concept_heads, config.enc_ff)
        self.decoder = TokenDecoder(config.d_token, config.d_concept, config.dec_layers, config.dec_heads, config.dec_ff)
        self.lm_head = nn.Linear(config.d_token, config.vocab_size, bias=False)
        
        # Output Scaling for muP (Eq. 21)
        self.output_scale = 1.0 / config.s_token

    def forward(self, input_ids: torch.Tensor, training: bool = True):
        """
        Args:
            input_ids: Tokenized input sequence [B, L]
            training: Flag to control boundary sampling strategy
        Returns:
            logits: Next-token predictions [B, L, vocab_size]
            b: Boundary indicators for L_aux loss computation [B, L]
            p: Boundary probabilities for L_aux loss computation [B, L]
        """
        # Stage 1: Encoding
        H = self.encoder(self.embedding(input_ids))
        
        # Stage 2: Dynamic Segmentation
        C, b, p = self.segmentation(H, training=training)
        
        # Compute segment lengths for the Decoder's Concept Replication
        segment_lengths = self._compute_segment_lengths(b, H.shape[1])
        
        # Stage 3: Concept Reasoning
        Z = self.concept_model(C)
        
        # Stage 4: Token-Level Decoding
        H_decoded = self.decoder(H, Z, segment_lengths)
        logits = self.lm_head(H_decoded) * self.output_scale
        
        return logits, b, p

    def _compute_segment_lengths(self, b: torch.Tensor, L: int) -> torch.Tensor:
        """Computes the number of tokens in each concept segment."""
        B, _ = b.shape
        segment_ids = (torch.cumsum(b, dim=1) - 1).long()
        max_M = int(segment_ids[:, -1].max().item()) + 1
        
        lengths = torch.zeros(B, max_M, device=b.device, dtype=torch.long)
        
        # segment_ids is already [B, L] (2D), which perfectly matches lengths.
        # Removed .unsqueeze(-1) and the extra dimension in torch.ones.
        idx = segment_ids 
        lengths.scatter_add_(1, idx, torch.ones(B, L, device=b.device, dtype=torch.long))
        
        return lengths
