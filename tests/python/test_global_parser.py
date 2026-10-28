# tests/python/test_global_parser.py
import torch
import torch.distributed as dist
import os
from dlcm.training.global_parser import GlobalParser

def test_global_parser():
    print(f"🧪 Testing Global Parser on Rank {os.environ.get('RANK', 'Unknown')}...")
    
    # Initialize dummy distributed environment (CPU-friendly "gloo" backend)
    dist.init_process_group(
        backend="gloo", 
        init_method="env://", 
        rank=int(os.environ["RANK"]), 
        world_size=int(os.environ["WORLD_SIZE"])
    )
    
    parser = GlobalParser(target_ratio_R=4.0)
    
    # Simulate local batch probabilities and boundaries
    p = torch.full((2, 100), 0.25) # Expected 25% boundaries
    b = torch.zeros((2, 100))
    b[:, ::4] = 1.0 # Actual 25% boundaries
    
    L_aux = parser.compute_auxiliary_loss(p, b)
    print(f"✅ Rank {dist.get_rank()} | L_aux: {L_aux.item():.4f}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_global_parser()
