import torch
import torch.nn as nn
import math
from typing import List, Dict, Any, Tuple

def initialize_mup_weights(model: nn.Module, d_base: int, d_token: int, d_concept: int):
    """
    Initializes model weights according to Decoupled µP (Section 6.1.1).
    
    Rule: Hidden linear weights W are initialized with variance σ²_base * s⁻¹, 
    where s is the width multiplier for that specific component.
    Embedding weights use fixed σ²_base.
    """
    s_token = d_token / d_base
    s_concept = d_concept / d_base
    
    # Base variance (standard normal initialization scaled)
    sigma_base = 1.0 
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Determine which component this linear layer belongs to
            if 'concept_model' in name:
                s = s_concept
            elif 'encoder' in name or 'decoder' in name:
                s = s_token
            else:
                # Fallback for shared or ambiguous layers (e.g., lm_head)
                s = s_token 
            
            # µP Initialization: σ² = σ²_base * s⁻¹
            std = sigma_base / math.sqrt(s)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # Embeddings use fixed variance (Section 6.1.1)
            nn.init.normal_(module.weight, mean=0.0, std=sigma_base)


def get_decoupled_mup_optimizer(
    model: nn.Module, 
    d_base: int, 
    d_token: int, 
    d_concept: int,
    lr_base_token: float = 1e-3,
    lr_base_concept: float = 1e-3,
    lr_base_others: float = 1e-3,
    weight_decay: float = 0.01,
    eps_base: float = 1e-8
) -> torch.optim.Optimizer:
    """
    Constructs an AdamW optimizer with decoupled learning rates and epsilons 
    based on the component's width multiplier (Section 6.1.1).
    """
    s_token = d_token / d_base
    s_concept = d_concept / d_base
    
    # Calculate effective LRs and Epsilons
    lr_token = lr_base_token / s_token
    lr_concept = lr_base_concept / s_concept
    lr_others = lr_base_others  # Biases, norms, and embeddings retain fixed LR
    
    eps_token = eps_base / s_token
    eps_concept = eps_base / s_concept
    eps_others = eps_base
    
    # Strictly mutually exclusive parameter grouping
    token_params = []
    concept_params = []
    other_params = [] 
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Categorize parameters using strict if/elif/else to prevent duplicates
        if 'concept_model' in name and 'bias' not in name and 'norm' not in name:
            concept_params.append(param)
        elif ('encoder' in name or 'decoder' in name or 'segmentation' in name) and 'bias' not in name and 'norm' not in name:
            token_params.append(param)
        else:
            # Biases, LayerNorms, Embeddings, and LM Head
            other_params.append(param)
            
    param_groups = []
    
    if len(token_params) > 0:
        param_groups.append({
            "params": token_params,
            "lr": lr_token,
            "eps": eps_token,
            "weight_decay": weight_decay,
            "name": "token_level_hidden"
        })
        
    if len(concept_params) > 0:
        param_groups.append({
            "params": concept_params,
            "lr": lr_concept,
            "eps": eps_concept,
            "weight_decay": weight_decay,
            "name": "concept_level_hidden"
        })
        
    if len(other_params) > 0:
        param_groups.append({
            "params": other_params,
            "lr": lr_others,
            "eps": eps_others,
            "weight_decay": 0.0, # Typically no weight decay for biases/norms/embeddings/lm_head
            "name": "others"
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer
