# tests/python/train_proxy.py
"""
87M Proxy Model Training Script for DLCM.
Matches the proxy configuration from Section 6.1.2 of the DLCM paper.
Uses a real streaming Hugging Face dataset to validate the Global Parser on actual text.
"""
import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

from dlcm.models.dlcm import DLCMConfig
from dlcm.training.trainer import DLCMTrainingModule

# =====================================================================
# Hugging Face Token Handling
# The bash script will inject this for Colab. If running locally, it will 
# fall back to python-dotenv to read your .env file, or remain empty.
# =====================================================================
if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"]:
    try:
        from dotenv import load_dotenv
        load_dotenv()  # Loads HF_TOKEN from a local .env file if it exists
    except ImportError:
        pass  # python-dotenv not installed, will proceed unauthenticated (rate-limited)


class StreamingTokenizedDataset(IterableDataset):
    """
    A lightweight streaming dataset that tokenizes text on-the-fly and yields 
    fixed-length sequences, mimicking the packed sequence training mentioned in Section 4.
    """
    def __init__(self, dataset_name: str, seq_len: int = 512, max_samples: int = 5000):
        self.seq_len = seq_len
        self.max_samples = max_samples
        
        # streaming=True prevents downloading the entire dataset to disk.
        # The datasets library will automatically use os.environ.get("HF_TOKEN") if present.
        self.dataset = load_dataset(dataset_name, "sample-10BT", split="train", streaming=True)
        
        # GPT-2 tokenizer is lightweight and perfect for proxy testing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        count = 0
        for item in self.dataset:
            if count >= self.max_samples:
                break
                
            text = item.get("text", "")
            if not isinstance(text, str) or len(text) < 20:
                continue  # Skip overly short or invalid snippets
                
            # Tokenize and truncate/pad to exact seq_len
            encoded = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.seq_len,
                padding="max_length",
                return_tensors="pt"
            )
            
            yield {"input_ids": encoded["input_ids"].squeeze(0)}
            count += 1


def train_proxy():
    print("🚀 Initializing 87M Proxy DLCM Training...")
    
    config = DLCMConfig(
        vocab_size=50257,
        d_token=512,
        d_concept=1024,
        d_scan=512,
        enc_layers=4, enc_heads=8, enc_ff=1024,
        concept_layers=6, concept_heads=16, concept_ff=2048,
        dec_layers=4, dec_heads=8, dec_ff=1024,
        alpha=1.0,
        threshold=0.5,
        s_token=1.0
    )
    
    muP_config = {
        'd_base': 512,
        'target_R': 4.0,
        'lambda_aux': 0.1,
        'lr_base_token': 3e-4,
        'lr_base_concept': 3e-4,
        'lr_base_others': 3e-4,
        'weight_decay': 0.01,
        'eps_base': 1e-8,
        'max_steps': 1000  # Run 1000 steps to observe loss compression convergence
    }
    
    model_instance = DLCMTrainingModule(config, muP_config)
    total_params = sum(p.numel() for p in model_instance.parameters())
    print(f"📊 Model Parameters: ~{total_params / 1e6:.1f}M")
    
    print("📚 Loading streaming dataset (HuggingFaceFW/fineweb-edu)...")
    dataset = StreamingTokenizedDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        seq_len=512,
        max_samples=5000  # CRITICAL FIX: 1000 steps * 4 batch_size = 4000 samples needed.
    )
    
    # num_workers=0 is crucial for streaming datasets in PyTorch Lightning
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)
    
    trainer = Trainer(
        max_steps=muP_config['max_steps'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )
    
    print("\n⏳ Entering trainer.fit()...")
    print("⚠️  NOTE: The first batch may take 2-5 minutes to download/tokenize due to HF Hub rate limits.")
    print("⏳ Please be patient. If it exceeds 10 minutes, press Ctrl+C.\n")
    
    trainer.fit(model_instance, dataloader)
    
    print("\n🎉 87M Proxy Training Completed Successfully!")
    print("💡 Next Step: Check if 'train/realized_compression_ratio' converged toward 4.0.")

    # =====================================================================
    # EXPORT WEIGHTS FOR RUST INFERENCE
    # =====================================================================
    print("\n💾 Exporting trained weights to Safetensors for Rust inference...")
    try:
        from safetensors.torch import save_file
        
        # Extract the raw DLCMModel state_dict (stripping Lightning wrappers)
        # and save it to a known path that the Rust inference binary will look for.
        weights_path = "/content/dlcm_proxy_trained.safetensors"
        save_file(model_instance.model.state_dict(), weights_path)
        print(f"✅ Weights successfully exported to {weights_path}")
    except ImportError:
        print("⚠️ safetensors not installed. Skipping weight export.")
    except Exception as e:
        print(f"⚠️ Failed to export weights: {e}")


if __name__ == "__main__":
    train_proxy()

    # This bypasses Python's atexit handlers and prevents the 
    # HuggingFace streaming dataset threads from hanging the process.
    import os
    os._exit(0) 
