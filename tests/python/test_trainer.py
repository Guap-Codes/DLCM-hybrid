# tests/python/test_trainer.py
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from dlcm.models.dlcm import DLCMConfig
from dlcm.training.trainer import DLCMTrainingModule

class DummyDataset(Dataset):
    def __len__(self): return 100
    def __getitem__(self, idx):
        return {"input_ids": torch.randint(0, 1000, (128,))}

def test_training_loop():
    print("🧪 Testing 10-Step Training Loop...")
    
    # CRITICAL: Dimensions divisible by num_heads
    config = DLCMConfig(
        d_token=256, 
        d_concept=512, 
        enc_layers=2, enc_heads=8, enc_ff=512,
        concept_layers=2, concept_heads=8, concept_ff=1024,
        dec_layers=2, dec_heads=8, dec_ff=512
    )
    muP_config = {
        'd_base': 256, 
        'target_R': 4.0, 
        'lambda_aux': 0.1,
        'lr_base_token': 1e-3, 
        'lr_base_concept': 1e-3, 
        'lr_base_others': 1e-3, 
        'max_steps': 10
    }
    
    model = DLCMTrainingModule(config, muP_config)
    dataloader = DataLoader(DummyDataset(), batch_size=4)
    
    # Force CPU accelerator and disable logger for maximum speed on Celeron
    trainer = Trainer(
        max_steps=10, 
        accelerator="gpu", 
        devices=1, 
        enable_checkpointing=False,
        logger=False 
    )
    trainer.fit(model, dataloader)
    print("✅ 10-step training loop completed successfully!")

if __name__ == "__main__":
    test_training_loop()
