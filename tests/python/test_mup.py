# tests/python/test_mup.py
import torch
import math
from dlcm.models.dlcm import DLCMModel, DLCMConfig
from dlcm.training.decoupled_mup import initialize_mup_weights, get_decoupled_mup_optimizer

def test_mup_initialization_and_lr_scaling():
    print("🧪 Testing Decoupled µP Initialization and LR Scaling...")
    
    # Use a small config for fast testing
    config = DLCMConfig(
        vocab_size=500,
        d_token=64,
        d_concept=128,
        d_scan=64,
        enc_layers=1, enc_heads=4, enc_ff=128,
        concept_layers=1, concept_heads=8, concept_ff=256,
        dec_layers=1, dec_heads=4, dec_ff=128,
        s_token=1.0
    )
    
    model = DLCMModel(config)
    d_base = 64
    s_token = config.d_token / d_base  # 1.0
    s_concept = config.d_concept / d_base  # 2.0
    
    # 1. Test Initialization Variance
    print("  Checking initialization variance scaling...")
    # Concept model hidden weights should have variance scaled by 1/s_concept = 0.5
    concept_weight = model.concept_model.layers[0].attn.W_q.weight
    emp_var_concept = concept_weight.var().item()
    emp_var_token = model.encoder.layers[0].attn.W_q.weight.var().item()
    
    # The ratio of variances should be roughly s_token / s_concept = 1.0 / 2.0 = 0.5
    var_ratio = emp_var_concept / emp_var_token
    assert 0.4 < var_ratio < 0.6, f"Variance ratio {var_ratio} not close to expected 0.5"
    print(f"  ✅ Variance ratio (concept/token): {var_ratio:.2f} (expected ~0.5)")

    # 2. Test Optimizer LR Scaling
    print("  Checking optimizer learning rate scaling...")
    lr_base_token = 1e-3
    lr_base_concept = 1e-3
    lr_base_others = 1e-3
    
    optimizer = get_decoupled_mup_optimizer(
        model=model,
        d_base=d_base,
        d_token=config.d_token,
        d_concept=config.d_concept,
        lr_base_token=lr_base_token,
        lr_base_concept=lr_base_concept,
        lr_base_others=lr_base_others
    )
    
    # Map parameter names to their expected LRs
    param_lrs = {}
    for group in optimizer.param_groups:
        lr = group['lr']
        for p in group['params']:
            for n, param in model.named_parameters():
                if param is p:
                    param_lrs[n] = lr
                    break
                    
    # Check a token-level hidden weight
    enc_wq_lr = param_lrs['encoder.layers.0.attn.W_q.weight']
    expected_token_lr = lr_base_token / s_token  # 1e-3 / 1.0 = 1e-3
    assert math.isclose(enc_wq_lr, expected_token_lr, rel_tol=1e-5), f"Encoder LR {enc_wq_lr} != {expected_token_lr}"
    
    # Check a concept-level hidden weight
    conc_wq_lr = param_lrs['concept_model.layers.0.attn.W_q.weight']
    expected_concept_lr = lr_base_concept / s_concept  # 1e-3 / 2.0 = 5e-4
    assert math.isclose(conc_wq_lr, expected_concept_lr, rel_tol=1e-5), f"Concept LR {conc_wq_lr} != {expected_concept_lr}"
    
    # Check an "other" weight (e.g., embedding)
    emb_lr = param_lrs['embedding.weight']
    assert math.isclose(emb_lr, lr_base_others, rel_tol=1e-5), f"Embedding LR {emb_lr} != {lr_base_others}"
    
    print(f"  ✅ Token LR: {enc_wq_lr:.1e} (expected {expected_token_lr:.1e})")
    print(f"  ✅ Concept LR: {conc_wq_lr:.1e} (expected {expected_concept_lr:.1e})")
    print(f"  ✅ Embedding LR: {emb_lr:.1e} (expected {lr_base_others:.1e})")

    # 3. Test Output Logit Scaling
    print("  Checking output logit scaling (1 / s_token)...")
    assert math.isclose(model.output_scale, 1.0 / s_token, rel_tol=1e-5), "Output scale mismatch"
    
    # Verify with a different s_token
    config2 = DLCMConfig(d_token=128, d_concept=128, d_scan=64, s_token=2.0) 
    model2 = DLCMModel(config2)
    assert math.isclose(model2.output_scale, 1.0 / 2.0, rel_tol=1e-5), "Output scale mismatch for s_token=2.0"
    print(f"  ✅ Output scale correctly set to 1/s_token")

    print("\n🎉 ALL µP TESTS PASSED! Decoupled scaling is mathematically sound.")

if __name__ == "__main__":
    test_mup_initialization_and_lr_scaling()
