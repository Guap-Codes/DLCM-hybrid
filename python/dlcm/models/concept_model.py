import torch
import torch.nn as nn
from .encoder import RMSNorm, EncoderLayer

class ConceptReasoningBackbone(nn.Module):
    """
    Stage 3: Concept-Level Reasoning (Section 3.4)
    A high-capacity Transformer that performs deep reasoning exclusively 
    on the compressed concept sequence C.
    """
    def __init__(self, d_concept: int, num_layers: int, num_heads: int, d_ff: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_concept, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_concept)
        
    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: Compressed concept representations [B, M, d_concept]
        Returns:
            Z: Enriched concept representations [B, M, d_concept]
        """
        for layer in self.layers:
            C = layer(C)
        return self.norm(C)
