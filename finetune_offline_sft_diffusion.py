import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import random
from tokenizer_pselfies import PSELFIESTokenizer
from model import ConditionalDiffusionLM 

# ============================================================================
# CONFIGURATION
# ============================================================================
PRETRAIN_CKPT = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_diffusion/checkpoints/pretrain//best_pretrain/best_pretrain.pt"
DATASET_PATH = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/Egc_pselfies_sa.csv" 
OUTPUT_DIR = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_diffusion/checkpoints/sft_diffusion"

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-5 

TARGET_MIN = 0.0 
TARGET_MAX = 10.0 
# ============================================================================

def apply_diffusion_masking(input_ids, mask_ratios, tokenizer, device, use_span_masking=True):
    masked_input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100) # -100 is ignored by CrossEntropyLoss
    
    is_special = (
        (input_ids == tokenizer.pad_id) |
        (input_ids == tokenizer.bos_id) |
        (input_ids == tokenizer.eos_id)
    )
    
    for i in range(input_ids.size(0)):
        valid_positions = (~is_special[i]).nonzero(as_tuple=True)[0]
        n_valid = len(valid_positions)
        n_to_mask = int(mask_ratios[i].item() * n_valid)
        
        if n_to_mask > 0:
            if use_span_masking:
                masked_indices_set = set()
                attempts = 0
                
                # Pre-sample span lengths to avoid CPU bottleneck in the loop
                span_length_samples = torch.poisson(
                    torch.full((200,), 3.0)
                ).clamp(1, 8).int().tolist()
                
                while len(masked_indices_set) < n_to_mask and attempts < 100:
                    # Index into pre-sampled list
                    span_len = span_length_samples[attempts % 200]
                    
                    # Sample start index within valid positions
                    if n_valid <= span_len:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, n_valid - span_len)
                    
                    # Add contiguous span to set (handles overlaps cleanly)
                    for offset in range(span_len):
                        if start_idx + offset < n_valid:
                            masked_indices_set.add(start_idx + offset)
                    
                    attempts += 1
                
                # Apply the accumulated spans
                for idx in masked_indices_set:
                    pos = valid_positions[idx].item()
                    labels[i, pos] = input_ids[i, pos]
                    masked_input_ids[i, pos] = tokenizer.mask_id
                    
            else:
                # ORIGINAL RANDOM MASKING (Fallback)
                perm = torch.randperm(n_valid, device=device)[:n_to_mask]
                positions_to_mask = valid_positions[perm]
                labels[i, positions_to_mask] = input_ids[i, positions_to_mask]
                masked_input_ids[i, positions_to_mask] = tokenizer.mask_id
            
    mask_counts = (masked_input_ids == tokenizer.mask_id).sum(dim=1).float()
    seq_lengths = (~is_special).sum(dim=1).float().clamp(min=1.0)
    timesteps = mask_counts / seq_lengths
    attn_mask = (masked_input_ids != tokenizer.pad_id).long()
    
    return masked_input_ids, labels, attn_mask, timesteps

class KhazanaSFTDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        
        self.sequences = []
        self.targets = []
        
        sequence_col = "pselfies" if "pselfies" in self.df.columns else self.df.columns[0]
        target_col = "Egc" if "Egc" in self.df.columns else self.df.columns[1]
        
        for sequence, bandgap in tqdm(zip(self.df[sequence_col], self.df[target_col]), total=len(self.df)):
            token_ids = self.tokenizer.encode(str(sequence), add_special_tokens=True)
            if len(token_ids) > self.tokenizer.max_length:
                continue
                
            self.sequences.append(token_ids)
            self.targets.append(float(bandgap))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        padded_seq = seq + [self.tokenizer.pad_id] * (self.tokenizer.max_length - len(seq))
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }

def train_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Starting Supervised Fine-Tuning (Diffusion) on {device}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = PSELFIESTokenizer.load("/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/mcts_diffusion/checkpoints/pretrain/best_pretrain/tokenizer.pt")
    
    model = ConditionalDiffusionLM(
        vocab_size=tokenizer.vocab_size,
        use_property_conditioning=True
    ).to(device)

    ckpt = torch.load(PRETRAIN_CKPT, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # UNFREEZE EVERYTHING FOR SFT
    model.train() 
    for param in model.parameters():
        param.requires_grad = True

    full_dataset = KhazanaSFTDataset(DATASET_PATH, tokenizer)
    
    generator = torch.Generator().manual_seed(42)
    val_size = int(0.10 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss() 
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)

    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train() 
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            raw_targets = batch['target'].to(device)
            target_norm = ((raw_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)).clamp(0.0, 1.0)
            
            # Mask anywhere from 10% to 100% of the sequence
            mask_ratios = torch.rand(input_ids.size(0), device=device) * 0.90 + 0.10
            masked_ids, labels, attn_mask, timesteps = apply_diffusion_masking(input_ids, mask_ratios, tokenizer, device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                # Forward pass - predict the masked tokens
                logits, _ = model(masked_ids, attention_mask=attn_mask, timestep=timesteps, prop=target_norm) 
                
                # Flatten and calculate loss ONLY on the -100 masked tokens
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({'Mask_Loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                raw_targets = batch['target'].to(device)
                target_norm = ((raw_targets - TARGET_MIN) / (TARGET_MAX - TARGET_MIN)).clamp(0.0, 1.0)
                
                # Evaluate on a consistent 50% mask ratio
                eval_ratios = torch.full((input_ids.size(0),), 0.5, device=device)
                masked_ids, labels, attn_mask, timesteps = apply_diffusion_masking(input_ids, eval_ratios, tokenizer, device)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits, _ = model(masked_ids, attention_mask=attn_mask, timestep=timesteps, prop=target_norm)
                    loss = criterion(logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))
                    total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"--- Epoch {epoch+1} Results ---")
        print(f"Train Mask Loss: {avg_train_loss:.4f}")
        print(f"Val Mask Loss  : {avg_val_loss:.4f}\n")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(), 
                'val_loss': avg_val_loss,
            }, f"{OUTPUT_DIR}/best_sft_model_diffusion.pt")
            
if __name__ == "__main__":
    train_sft()