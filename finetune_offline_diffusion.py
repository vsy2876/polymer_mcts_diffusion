import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import math
from tqdm import tqdm
import random

from tokenizer_pselfies import PSELFIESTokenizer
from model import ConditionalDiffusionLM 

# ============================================================================
# CONFIGURATION
# ============================================================================
PRETRAIN_CKPT = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_diffusion/checkpoints/sft_diffusion/best_sft_model_diffusion.pt"
DATASET_PATH = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/Egc_pselfies_sa.csv" 
OUTPUT_DIR = "./checkpoints/offline_finetune_diffusion"

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4

TARGET_MIN = 0.0 
TARGET_MAX = 10.0 
OFFLINE_SIGMA = 0.5 
# ============================================================================

def apply_diffusion_masking(input_ids, mask_ratios, tokenizer, device):
    masked_input_ids = input_ids.clone()
    is_special = (
        (input_ids == tokenizer.pad_id) |
        (input_ids == tokenizer.bos_id) |
        (input_ids == tokenizer.eos_id)
    )
    
    for i in range(input_ids.size(0)):
        valid_positions = (~is_special[i]).nonzero(as_tuple=True)[0]
        n_to_mask = int(mask_ratios[i].item() * len(valid_positions))
        if n_to_mask > 0:
            perm = torch.randperm(len(valid_positions), device=device)[:n_to_mask]
            positions_to_mask = valid_positions[perm]
            masked_input_ids[i, positions_to_mask] = tokenizer.mask_id
            
    mask_counts = (masked_input_ids == tokenizer.mask_id).sum(dim=1).float()
    seq_lengths = (~is_special).sum(dim=1).float().clamp(min=1.0)
    timesteps = mask_counts / seq_lengths
    attn_mask = (masked_input_ids != tokenizer.pad_id).long()
    
    return masked_input_ids, attn_mask, timesteps

class PolymerGenomeDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        
        self.sequences = []
        self.true_bandgaps = []
        self.sa_scores = []
        
        sequence_col = "pselfies" 
        target_col = "Egc" 
        sa_col = "sa_score" 
        
        if sa_col not in self.df.columns:
            raise ValueError(f"CRITICAL: '{sa_col}' not found in {csv_path}. Please generate the SA scores first!")
            
        for sequence, bandgap, sa in tqdm(zip(self.df[sequence_col], self.df[target_col], self.df[sa_col]), total=len(self.df)):
            token_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
            if len(token_ids) > self.tokenizer.max_length:
                continue
                
            self.sequences.append(token_ids)
            self.true_bandgaps.append(float(bandgap))
            self.sa_scores.append(float(sa))

        print(f"Loaded {len(self.sequences)} valid periodic polymers.")
        
        min_gap = min(self.true_bandgaps)
        max_gap = max(self.true_bandgaps)
        print(f"Physical Egc Range: {min_gap:.3f} eV -> {max_gap:.3f} eV")
        assert max_gap <= TARGET_MAX + 1e-6, f"FATAL: Dataset contains bandgaps up to {max_gap:.2f} eV but TARGET_MAX={TARGET_MAX}."
        assert min_gap >= TARGET_MIN - 1e-6, f"FATAL: Dataset contains bandgaps down to {min_gap:.2f} eV but TARGET_MIN={TARGET_MIN}."

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = seq + [self.tokenizer.pad_id] * (self.tokenizer.max_length - len(seq))
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.long),
            "true_bandgap": torch.tensor(self.true_bandgaps[idx], dtype=torch.float32),
            "sa_score": torch.tensor(self.sa_scores[idx], dtype=torch.float32)
        }

def sanity_check(model, tokenizer, device):
    print("--- Running Architecture Sanity Check ---")
    model.eval()
    with torch.no_grad():
        dummy_ids = torch.randint(0, tokenizer.vocab_size, (4, tokenizer.max_length), device=device)
        dummy_mask = torch.ones_like(dummy_ids)
        dummy_time = torch.rand(4, device=device)
        dummy_prop = torch.rand(4, device=device)
        
        out = model(input_ids=dummy_ids, attention_mask=dummy_mask, timestep=dummy_time, prop=dummy_prop)
        
        assert isinstance(out, tuple), "Model must return a tuple of (logits, values)"
        assert len(out) == 2, "Model must return exactly two elements"
        
        logits, values = out
        assert values.dim() in [1, 2], f"FATAL: Expected sequence-level scalar (B,) or (B,1), got {values.shape}."
    print("✓ Sanity Check Passed! Model outputs are safe.")

def train_offline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Training Masked Diffusion Critic on {device} (AMP Enabled: {use_amp})")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = PSELFIESTokenizer.load("checkpoints/pretrain/best_pretrain/tokenizer.pt")
    
    assert hasattr(tokenizer, 'mask_id') and tokenizer.mask_id is not None, "FATAL: Tokenizer has no mask_id!"
    
    model = ConditionalDiffusionLM(
        vocab_size=tokenizer.vocab_size,
        use_property_conditioning=True  # <--- CRITICAL: Wakes up the AdaLN layers!
    ).to(device)

    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    sanity_check(model, tokenizer, device)

    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
        
    trainable_params = []
    print("\n--- Active Trainable Parameters ---")
    for name, param in model.named_parameters():
        if any(key in name for key in ['value_head', 'property_embedding']):
            param.requires_grad = True
            trainable_params.append(param)
            print(f"  - {name}: {param.numel():,} params")

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total Trainable Parameters: {total_trainable:,}\n")
    assert total_trainable > 0, "FATAL: No trainable parameters found!"

    full_dataset = PolymerGenomeDataset(DATASET_PATH, tokenizer)
    
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)

    best_combined_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.eval() 
        for name, module in model.named_modules():
            if any(key in name for key in ['value_head', 'property_embedding']):
                module.train()
                
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            true_bandgaps = batch['true_bandgap'].to(device)
            sa_scores = batch['sa_score'].to(device)
            
            mask = torch.rand_like(true_bandgaps) < 0.3
            local_targets = true_bandgaps + torch.randn_like(true_bandgaps) * 0.25
            global_targets = torch.rand_like(true_bandgaps) * (TARGET_MAX - TARGET_MIN) + TARGET_MIN
            sampled_targets = torch.where(mask, local_targets, global_targets).clamp(TARGET_MIN, TARGET_MAX)
            
            # --- THE MISSING LINE RESTORED ---
            target_norm = (sampled_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
            
            # 1. Physics Reward
            physics_reward = torch.exp(-((true_bandgaps - sampled_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
            
            # 2. SA Penalty (Matched to MCTS continuous formula)
            sa_penalty_factor = torch.exp(-0.5 * torch.clamp(sa_scores - 3.0, min=0.0))
            
            # 3. Combined Continuous Reward
            combined_reward = physics_reward * sa_penalty_factor
            
            # 4. Final RL Reward with -0.5 MCTS Floor
            expected_rl_reward = (combined_reward * 1.3) - 0.3

            mask_ratios = torch.rand(input_ids.size(0), device=device) * 0.90 
            masked_ids, attn_mask, timesteps = apply_diffusion_masking(input_ids, mask_ratios, tokenizer, device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, values = model(
                    input_ids=masked_ids, 
                    attention_mask=attn_mask, 
                    timestep=timesteps, 
                    prop=target_norm # Now perfectly defined
                ) 
                
                if values.dim() == 2 and values.shape[1] == 1:
                    predicted_values = values.squeeze(-1)
                elif values.dim() == 1:
                    predicted_values = values
                else:
                    raise ValueError(f"CRITICAL: Value Head returned unexpected token-level shape {values.shape}")
                
                assert predicted_values.shape == expected_rl_reward.shape, f"FATAL Shape mismatch: {predicted_values.shape} vs {expected_rl_reward.shape}"
                
                loss = criterion(predicted_values, expected_rl_reward)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item():.4f}", 'LR': f"{scheduler.get_last_lr()[0]:.2e}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION LOOPS (SA PENALTY INTEGRATED) ---
        model.eval()
        total_perf_loss, total_mask_loss, total_med_loss, total_worst_loss = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                true_bandgaps = batch['true_bandgap'].to(device)
                sa_scores = batch['sa_score'].to(device)
                # 2. SA Penalty (Matched to MCTS continuous formula)
                sa_penalty_factor = torch.exp(-0.5 * torch.clamp(sa_scores - 3.0, min=0.0))

                # 1. PERFECT MATCH
                perf_targets = true_bandgaps.clone().clamp(TARGET_MIN, TARGET_MAX)
                perf_norm = (perf_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                perf_physics = torch.ones_like(true_bandgaps) 
                
                combined_perf = perf_physics * sa_penalty_factor
                perf_expected = (combined_perf * 1.3) - 0.3
                
                clean_ratios = torch.zeros(input_ids.size(0), device=device)
                clean_ids, clean_mask, clean_time = apply_diffusion_masking(input_ids, clean_ratios, tokenizer, device)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, perf_values = model(clean_ids, attention_mask=clean_mask, timestep=clean_time, prop=perf_norm)
                    perf_preds = perf_values.squeeze(-1) if perf_values.dim() == 2 else perf_values
                    total_perf_loss += criterion(perf_preds, perf_expected).item()

                # 2. 50% MASKED PERFECT MATCH
                half_ratios = torch.full((input_ids.size(0),), 0.5, device=device)
                half_ids, half_mask, half_time = apply_diffusion_masking(input_ids, half_ratios, tokenizer, device)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, mask_values = model(half_ids, attention_mask=half_mask, timestep=half_time, prop=perf_norm)
                    mask_preds = mask_values.squeeze(-1) if mask_values.dim() == 2 else mask_values
                    total_mask_loss += criterion(mask_preds, perf_expected).item()

                # 3. MEDIUM MATCH (+0.5 eV)
                med_targets = (true_bandgaps + OFFLINE_SIGMA).clamp(TARGET_MIN, TARGET_MAX)
                med_norm = (med_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                med_physics = torch.exp(-((true_bandgaps - med_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
                
                combined_med = med_physics * sa_penalty_factor
                med_expected = (combined_med * 1.3) - 0.3
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, med_values = model(clean_ids, attention_mask=clean_mask, timestep=clean_time, prop=med_norm)
                    med_preds = med_values.squeeze(-1) if med_values.dim() == 2 else med_values
                    total_med_loss += criterion(med_preds, med_expected).item()
                
                # 4. WORST-CASE MISMATCH
                midpoint = (TARGET_MIN + TARGET_MAX) / 2.0
                worst_targets = torch.where(
                    true_bandgaps < midpoint,
                    torch.full_like(true_bandgaps, TARGET_MAX),
                    torch.full_like(true_bandgaps, TARGET_MIN)
                )
                worst_norm = (worst_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)
                worst_physics = torch.exp(-((true_bandgaps - worst_targets) ** 2) / (2 * OFFLINE_SIGMA ** 2))
                
                combined_worst = worst_physics * sa_penalty_factor
                worst_expected = (combined_worst * 1.3) - 0.3
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    _, worst_values = model(clean_ids, attention_mask=clean_mask, timestep=clean_time, prop=worst_norm)
                    worst_preds = worst_values.squeeze(-1) if worst_values.dim() == 2 else worst_values
                    total_worst_loss += criterion(worst_preds, worst_expected).item()
                
        avg_perf_loss = total_perf_loss / len(val_loader)
        avg_mask_loss = total_mask_loss / len(val_loader)
        avg_med_loss = total_med_loss / len(val_loader)
        avg_worst_loss = total_worst_loss / len(val_loader)
        
        combined_val_loss = (avg_perf_loss + avg_med_loss + avg_worst_loss) / 3.0
        strict_val_loss = max(avg_perf_loss, avg_med_loss, avg_worst_loss)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train MSE    : {avg_train_loss:.4f}")
        print(f"Val Perfect  : {avg_perf_loss:.4f} (Clean) | {avg_mask_loss:.4f} (50% Masked, diagnostic)")
        print(f"Val Medium   : {avg_med_loss:.4f}")
        print(f"Val Worst    : {avg_worst_loss:.4f}")
        print(f"Combined Val : {combined_val_loss:.4f}")
        print(f"Strict Val   : {strict_val_loss:.4f} (Diagnostic)\n")
        
        if combined_val_loss < best_combined_loss:
            best_combined_loss = combined_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'combined_val_loss': combined_val_loss,
                'strict_val_loss': strict_val_loss,
                'target_min': TARGET_MIN,
                'target_max': TARGET_MAX,
                'offline_sigma': OFFLINE_SIGMA,
            }, f"{OUTPUT_DIR}/best_offline_surrogate_diffusion.pt")
            
if __name__ == "__main__":
    train_offline()