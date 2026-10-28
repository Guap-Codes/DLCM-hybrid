import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict

# Import local modules
from dlcm.models.dlcm import DLCMModel, DLCMConfig
from dlcm.training.global_parser import GlobalParser
from dlcm.training.decoupled_mup import initialize_mup_weights, get_decoupled_mup_optimizer

class DLCMTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training the Dynamic Large Concept Model.
    Integrates the 4-stage pipeline, Global Parser (L_aux), and Decoupled µP.
    """
    
    def __init__(self, config: DLCMConfig, muP_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.muP_config = muP_config
        
        # 1. Initialize Model
        self.model = DLCMModel(config)
        
        # 2. Apply µP Initialization (Section 6.1.1)
        initialize_mup_weights(
            self.model, 
            d_base=muP_config['d_base'],
            d_token=config.d_token,
            d_concept=config.d_concept
        )
        
        # 3. Initialize Global Parser for Adaptive Compression (Section 3.3.3)
        self.global_parser = GlobalParser(target_ratio_R=muP_config.get('target_R', 4.0))
        self.lambda_aux = muP_config.get('lambda_aux', 0.1)
        
        # Metric tracking
        self.train_loss_avg = 0.0
        self.train_count = 0

    def forward(self, input_ids: torch.Tensor, training: bool = True):
        return self.model(input_ids, training=training)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        
        # Forward Pass (Training=True enables Bernoulli boundary sampling)
        logits, b, p = self.model(input_ids, training=True)
        
        # 1. Compute Cross-Entropy Loss (L_CE)
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        L_CE = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # 2. Compute Auxiliary Loss (L_aux) via Global Parser
        # Note: We slice p and b to match the prediction length (L-1)
        L_aux = self.global_parser.compute_auxiliary_loss(
            p=p[..., :-1], 
            b=b[..., :-1], 
            lambda_aux=self.lambda_aux
        )
        
        # 3. Total Loss (Eq. 15)
        loss = L_CE + L_aux
        
        # Logging
        self.log("train/loss_total", loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_CE", L_CE, sync_dist=True)
        self.log("train/loss_aux", L_aux, sync_dist=True)
        
        # Monitor realized compression ratio
        realized_R = self.global_parser.get_realized_compression_ratio(b)
        self.log("train/realized_compression_ratio", realized_R, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        input_ids = batch["input_ids"]
        
        # Inference uses hard thresholding (training=False)
        logits, b, p = self.model(input_ids, training=False)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        L_CE = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        L_aux = self.global_parser.compute_auxiliary_loss(
            p=p[..., :-1], b=b[..., :-1], lambda_aux=self.lambda_aux
        )
        
        loss = L_CE + L_aux
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        
        realized_R = self.global_parser.get_realized_compression_ratio(b)
        self.log("val/realized_compression_ratio", realized_R, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configures the Decoupled µP Optimizer (Section 6.1).
        """
        optimizer = get_decoupled_mup_optimizer(
            model=self.model,
            d_base=self.muP_config['d_base'],
            d_token=self.config.d_token,
            d_concept=self.config.d_concept,
            lr_base_token=self.muP_config['lr_base_token'],
            lr_base_concept=self.muP_config['lr_base_concept'],
            lr_base_others=self.muP_config['lr_base_others'],
            weight_decay=self.muP_config.get('weight_decay', 0.01),
            eps_base=self.muP_config.get('eps_base', 1e-8)
        )
        
        # Optional: Add a learning rate scheduler (e.g., Cosine Decay with Warmup)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.muP_config.get('max_steps', 100000),
            eta_min=1e-5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# ==========================================
# Example Usage / Entry Point
# ==========================================
if __name__ == "__main__":
    from pytorch_lightning.strategies import DDPStrategy
    
    # 1. Define Configurations
    dlcm_config = DLCMConfig(
        d_token=1536,
        d_concept=3072,
        # ... other params from Table 3
    )
    
    muP_config = {
        'd_base': 1536,          # Base width for µP scaling
        'target_R': 4.0,         # Target compression ratio (Section 6.3.1)
        'lambda_aux': 0.1,       # Weight for L_aux (Eq. 15)
        'lr_base_token': 3e-4,   # Tuned on 87M proxy (Section 6.1.2)
        'lr_base_concept': 3e-4, # Tuned on 87M proxy (Section 6.1.2)
        'lr_base_others': 3e-4,
        'max_steps': 100000
    }
    
    # 2. Initialize Module
    model = DLCMTrainingModule(dlcm_config, muP_config)
    
    # 3. Setup Trainer (using DDP for Global Parser AllReduce)
    trainer = pl.Trainer(
        max_steps=muP_config['max_steps'],
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=False), # DLCM has dynamic graphs if not careful, but our impl is static graph friendly
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        # accumulate_grad_batches=4 # Simulating large batch for Global Parser stability
    )
    
    # 4. Train (Assuming dummy_data_module is defined)
    # trainer.fit(model, datamodule=dummy_data_module)
    print("Trainer configured successfully with Decoupled µP and Global Parser.")
