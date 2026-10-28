# tests/python/nano_test_trainer.py
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from dlcm.models.dlcm import DLCMConfig
from dlcm.training.trainer import DLCMTrainingModule

class DummyDataset(Dataset):
    def __len__(self): return 20
    def __getitem__(self, idx):
        # Tiny sequence length to save RAM
        return {"input_ids": torch.randint(0, 500, (32,))}

def test_nano_training_loop():
    print("🧪 Testing Nano Training Loop (Optimized for low RAM)...")
    
    # NANO CONFIG: ~0.5 Million parameters (Safe for Celeron)
    config = DLCMConfig(
        vocab_size=500,
        d_token=64, 
        d_concept=128, 
        d_scan=64,
        enc_layers=1, enc_heads=4, enc_ff=128,
        concept_layers=1, concept_heads=8, concept_ff=256,
        dec_layers=1, dec_heads=4, dec_ff=128
    )
    
    muP_config = {
        'd_base': 64, 
        'target_R': 4.0, 
        'lambda_aux': 0.1,
        'lr_base_token': 1e-3, 
        'lr_base_concept': 1e-3, 
        'lr_base_others': 1e-3, 
        'max_steps': 5  # Just 5 steps to prove it works
    }
    
    model = DLCMTrainingModule(config, muP_config)
    
    # Tiny batch size to minimize activation memory
    dataloader = DataLoader(DummyDataset(), batch_size=2, num_workers=0)
    
    # Force CPU, disable logging/checkpointing to save every possible MB of RAM
    trainer = Trainer(
        max_steps=5, 
        accelerator="cpu", 
        devices=1, 
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True
    )
    
    print("🚀 Starting training (this should take ~10 seconds)...")
    trainer.fit(model, dataloader)
    print("✅ 5-step nano training loop completed successfully without OOM!")

if __name__ == "__main__":
    test_nano_training_loop()
