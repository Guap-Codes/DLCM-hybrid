import torch
import torch.distributed as dist
from typing import Tuple

class GlobalParser:
    """
    Implements the Global Parser for Adaptive Compression (Section 3.3.3 & 8.2).
    
    Unlike per-sequence regularization which causes optimization conflicts (Section 8.1),
    the Global Parser enforces the compression ratio at the batch level across all 
    distributed ranks, allowing the model to adaptively allocate compression budgets 
    based on local information density.
    """
    
    def __init__(self, target_ratio_R: float = 4.0):
        """
        Args:
            target_ratio_R (float): The target compression ratio (e.g., R=4 means 4 tokens per concept).
        """
        self.R = target_ratio_R

    def compute_auxiliary_loss(
        self, 
        p: torch.Tensor, 
        b: torch.Tensor, 
        lambda_aux: float = 1.0
    ) -> torch.Tensor:
        """
        Computes the global auxiliary loss L_aux (Eq. 10).
        
        Args:
            p (torch.Tensor): Boundary probabilities [B, L]. Continuous values in [0, 1].
            b (torch.Tensor): Discrete boundary indicators [B, L]. Binary values {0, 1}.
            lambda_aux (float): Weight for the auxiliary loss (Eq. 15: L = L_CE + λ L_aux).
            
        Returns:
            torch.Tensor: Scalar auxiliary loss tensor.
        """
        # 1. Compute local sums and token counts
        # G_local: sum of expected boundary probabilities (Eq. 8 numerator part)
        # F_local: sum of actual boundary indicators (Eq. 9 numerator part)
        # T_local: total number of tokens in this micro-batch (Eq. 8/9 denominator)
        
        G_local = p.sum()
        F_local = b.sum()
        T_local = torch.tensor(p.numel(), device=p.device, dtype=p.dtype)
        
        # 2. Synchronize across distributed ranks using AllReduce
        if dist.is_available() and dist.is_initialized():
            # Sum across all GPUs
            dist.all_reduce(G_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(F_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(T_local, op=dist.ReduceOp.SUM)
            
        # 3. Compute global rates (Eq. 8 & 9)
        # Add epsilon to prevent division by zero in edge cases
        eps = 1e-8
        G_global = G_local / (T_local + eps)
        F_global = F_local / (T_local + eps)
        
        # 4. Compute Auxiliary Loss (Eq. 10)
        # L_aux = (R / (R - 1)) * [ (R - 1) * F_global * G_global + (1 - F_global) * (1 - G_global) ] - 1
        R = self.R
        
        term1 = (R - 1.0) * F_global * G_global
        term2 = (1.0 - F_global) * (1.0 - G_global)
        
        L_aux = (R / (R - 1.0)) * (term1 + term2) - 1.0
        
        return lambda_aux * L_aux

    def get_realized_compression_ratio(self, b: torch.Tensor) -> float:
        """
        Utility to monitor the actual compression ratio during training.
        Realized Ratio = Total Tokens / Total Concepts = L / M
        Since M = sum(b), Ratio = Total Tokens / sum(b)
        """
        T_local = torch.tensor(p.numel(), device=b.device, dtype=torch.float32) if 'p' in locals() else torch.tensor(b.numel(), device=b.device, dtype=torch.float32)
        F_local = b.sum()
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(T_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(F_local, op=dist.ReduceOp.SUM)
            
        return (T_local / (F_local + 1e-8)).item()
