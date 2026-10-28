# python/dlcm/evaluation/eval_harness.py
"""
Standalone Zero-Shot Evaluation Harness for DLCM.
Evaluates the model on multiple-choice reasoning tasks (e.g., HellaSwag, ARC).
"""
import torch
import json
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict
from dlcm.models.dlcm import DLCMModel, DLCMConfig

class ZeroShotEvaluator:
    def __init__(self, model: DLCMModel, tokenizer, device: str = "cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def _get_loglikelihood(self, context: str, continuation: str) -> float:
        """Computes the average log-likelihood of the continuation given the context."""
        # Tokenize context and continuation
        context_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        full_text = context + continuation
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)
        
        context_len = context_ids.shape[1]
        
        # Forward pass (Inference mode -> hard thresholding)
        logits, _, _ = self.model(full_ids, training=False)
        
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_ids[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # Extract log probs only for the continuation tokens
        cont_log_probs = log_probs[:, context_len-1:, :]
        cont_labels = shift_labels[:, context_len-1:]
        
        # Gather the log probs of the actual tokens
        target_log_probs = cont_log_probs.gather(2, cont_labels.unsqueeze(-1)).squeeze(-1)
        
        return target_log_probs.sum().item()

    def evaluate_multiple_choice(self, dataset_name: str, split: str = "validation", max_samples: int = 100) -> float:
        """
        Evaluates on a multiple-choice dataset (e.g., 'piqa', 'hellaswag', 'arc_easy').
        """
        print(f"\n🚀 Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        correct = 0
        total = 0
        
        # Note: Prompt formatting depends on the specific dataset. 
        # This is a generalized template for standard MCQ tasks.
        for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {dataset_name}")):
            if i >= max_samples:
                break
                
            # Extract question and choices (Adapt keys based on dataset schema)
            question = sample.get('question', sample.get('goal', ''))
            choices = sample.get('choices', {}).get('text', sample.get('endings', []))
            label = sample.get('label', sample.get('answerKey', 0))
            
            # Handle specific dataset quirks
            if dataset_name == 'hellaswag':
                question = sample['ctx']
                choices = sample['endings']
                label = int(sample['label'])
            elif dataset_name == 'piqa':
                question = sample['goal']
                choices = [sample['sol1'], sample['sol2']]
                label = int(sample['label'])

            if not choices:
                continue

            # Compute log-likelihood for each choice
            log_likelihoods = []
            for choice in choices:
                # Simple prompt format
                context = f"Question: {question}\nAnswer:"
                continuation = f" {choice}"
                ll = self._get_loglikelihood(context, continuation)
                log_likelihoods.append(ll)
            
            # Predict the choice with the highest log-likelihood
            pred = log_likelihoods.index(max(log_likelihoods))
            
            if pred == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"✅ {dataset_name} Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy

if __name__ == "__main__":
    # Example Usage
    from transformers import AutoTokenizer
    
    # 1. Load Tokenizer (Using DeepSeek-v3 tokenizer as per paper Section 5)
    # Note: Replace with actual DeepSeek tokenizer path
    tokenizer = AutoTokenizer.from_pretrained("gpt2") 
    
    # 2. Initialize Model
    config = DLCMConfig(d_token=512, d_concept=1024, enc_layers=4, concept_layers=8, dec_layers=2)
    model = DLCMModel(config)
    
    # 3. Run Evaluation
    evaluator = ZeroShotEvaluator(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate on PIQA (Physical Interaction QA)
    evaluator.evaluate_multiple_choice("piqa", max_samples=50)
