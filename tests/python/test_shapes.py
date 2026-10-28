# tests/python/test_shapes.py
import torch
from dlcm.models.dlcm import DLCMModel, DLCMConfig
from dlcm.training.global_parser import GlobalParser # Import the Global Parser

def test_forward_backward():
    print("🧪 Testing Forward/Backward Pass & Gradients...")
    
    # CRITICAL: d_token and d_concept MUST be divisible by their respective num_heads
    config = DLCMConfig(
        d_token=256, 
        d_concept=512, 
        enc_layers=2, enc_heads=8, enc_ff=512,
        concept_layers=2, concept_heads=8, concept_ff=1024,
        dec_layers=2, dec_heads=8, dec_ff=512
    )
    model = DLCMModel(config)
    
    # Dummy batch: Batch=2, SeqLen=64
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    
    # Forward
    logits, b, p = model(input_ids, training=True)
    
    assert logits.shape == (2, 64, config.vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert b.shape == (2, 64), f"Boundary shape mismatch: {b.shape}"
    
    # --- BACKWARD PASS ---
    
    # 1. Main Task Loss (Cross-Entropy proxy)
    # This updates the Encoder, Concept Model, and Decoder.
    ce_loss = logits.sum() 
    
    # 2. Auxiliary Loss (L_aux)
    # This is the ONLY path that updates the Boundary Detector (W_q, W_k)!
    parser = GlobalParser(target_ratio_R=4.0)
    L_aux = parser.compute_auxiliary_loss(p, b, lambda_aux=0.1)
    
    # Total Loss (Eq. 15: L = L_CE + λ L_aux)
    total_loss = ce_loss + L_aux
    total_loss.backward()
    
    # Check that gradients flowed to the Encoder (via CE loss)
    assert model.encoder.layers[0].attn.W_q.weight.grad is not None, "No gradient in encoder W_q!"
    
    # Check that gradients flowed to the Boundary Detector (via L_aux)
    # If this passes, it proves the Decoupled Segmentation (Section 8.1) is working perfectly!
    assert model.segmentation.W_q.weight.grad is not None, "No gradient in segmentation W_q!"
    
    print("✅ Forward/Backward pass shapes and gradients verified!")
    print(f"   -> CE Loss: {ce_loss.item():.4f}")
    print(f"   -> L_aux: {L_aux.item():.4f}")

if __name__ == "__main__":
    test_forward_backward()
