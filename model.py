# model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import numpy as np
import math

class GaussianFourierProjection(nn.Module):
    """Expands a single scalar into a rich frequency embedding."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Freeze weights
        self.W = nn.Parameter(torch.randn(1, embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        # x is (batch, 1)
        x_proj = x * self.W.to(x.device) * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ConditionalDiffusionLM(nn.Module):
    """Transformer-based discrete diffusion language model for conditional SMILES generation."""
    
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", 
                 vocab_size: int = None, 
                 use_property_conditioning: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        # Load pretrained transformer backbone
        config = AutoConfig.from_pretrained(model_name)
        if vocab_size is not None:
            config.vocab_size = vocab_size
            # Set pad_token_id to 0 (assuming your tokenizer uses 0 for [PAD])
            config.pad_token_id = 0

        self.backbone = AutoModel.from_pretrained(
            model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
        self.hidden_size = config.hidden_size
        
        # Property conditioning (for Egc, Tg, etc.)
        self.use_property_conditioning = use_property_conditioning
        if use_property_conditioning:
            self.property_embedding = nn.Sequential(
                GaussianFourierProjection(128),  # Transforms (batch, 1) -> (batch, 128)
                nn.Linear(128, self.hidden_size),
                nn.SiLU(), # SiLU is generally better than ReLU for diffusion
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        
        # Timestep (noise level) embedding for diffusion process
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, self.hidden_size)
        )
        
        # Output head for token prediction
        self.output_head = nn.Linear(self.hidden_size, vocab_size)
        
        # ── NEW ALPHA ZERO ADDITION: VALUE HEAD (THE CRITIC) ────────────────
        # This head evaluates incomplete, partially masked sequences during MCTS
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Tanh()  # Bounds the predicted sequence reward between -1.0 and 1.0
        )
        # ────────────────────────────────────────────────────────────────────
        
    def forward(self, input_ids, attention_mask, timestep, prop=None):
        """
        Forward pass through the diffusion model.
        """
        batch_size, seq_len = input_ids.shape
        
        # Get transformer embeddings
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Add timestep conditioning (broadcast to all positions)
        t_embed = self.timestep_embedding(timestep.unsqueeze(-1))  # (batch, hidden_size)
        hidden_states = hidden_states + t_embed.unsqueeze(1)  # (batch, seq_len, hidden_size)
        
       # Add property conditioning if available
        if self.use_property_conditioning and prop is not None:
            # Create a broadcast column vector tracking our shapes
            prop_tensor = prop.view(-1, 1)
            
            # Formulate the frequency embedding layout
            prop_embed = self.property_embedding(prop_tensor)  # (batch, hidden_size)
            
            # FIX: Execute masking unconditionally to avoid python control-flow graph breaks
            null_mask = (prop_tensor == -1.0)
            prop_embed = prop_embed.masked_fill(null_mask, 0.0)
                
            hidden_states = hidden_states + prop_embed.unsqueeze(1)  # (batch, seq_len, hidden_size)
        
        # Predict tokens for all positions (Policy Distribution)
        logits = self.output_head(hidden_states)  # (batch, seq_len, vocab_size)
        
        # ── NEW ALPHA ZERO ADDITION: MEAN-POOLED VALUE PREDICTION ───────────
        # Pool across sequence length using the attention mask to isolate valid tokens
        pool_mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * pool_mask).sum(dim=1)
        num_tokens = pool_mask.sum(dim=1).clamp(min=1.0)
        pooled_states = sum_embeddings / num_tokens
        
        # Compute the expected continuous reward value for this state
        values = self.value_head(pooled_states).squeeze(-1) # (batch,)
        # ────────────────────────────────────────────────────────────────────
        
        # ── COMMENTED OUT OLD RETURN ────────────────────────────────────────
        # return logits
        # ────────────────────────────────────────────────────────────────────
        
        # Return both the policy distributions and the continuous state values
        return logits, values