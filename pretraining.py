#!/usr/bin/env python
"""
POLYMER PSELFIES DIFFUSION MODEL - PRETRAINING SCRIPT
Unsupervised masked language modeling to teach the ModernBERT backbone PSELFIES grammar.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import signal
import sys

warnings.filterwarnings('ignore')

# Import custom modules
from model import ConditionalDiffusionLM
# ── CRITICAL EDIT 1: Swapped to PSELFIES Tokenizer ─────────────────────────
from tokenizer_pselfies import PSELFIESTokenizer
# ───────────────────────────────────────────────────────────────────────────

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/storage/home/hcoda1/2/vyadav68/r-cdeo3-0/polymers/PI1M_v2_pselfies.csv"
# Split Tracking Paths
OUTPUT_SCRATCH_DIR = '/storage/home/hcoda1/2/vyadav68/scratch/polymers/mcts_diffusion/checkpoints/pretrain'
OUTPUT_HOME_DIR    = './checkpoints/pretrain'
PLOT_DIR           = './plots/pretrain'

MODEL_NAME = "answerdotai/ModernBERT-base"
USE_PROPERTY_CONDITIONING = True
MAX_LENGTH = 128  # Adjusted to match your PSELFIES config

EPOCHS = 5
BATCH_SIZE = 256  
GRADIENT_ACCUMULATION_STEPS = 2  
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 1000
NUM_WORKERS = 8  

USE_AMP = True  
USE_COMPILE = True  
RESUME_FROM_STEP = 6000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# SPAN MASKING FUNCTION
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
                
                span_length_samples = torch.poisson(
                    torch.full((200,), 3.0)
                ).clamp(1, 8).int().tolist()
                
                while len(masked_indices_set) < n_to_mask and attempts < 100:
                    span_len = span_length_samples[attempts % 200]
                    
                    if n_valid <= span_len:
                        start_idx = 0
                    else:
                        start_idx = random.randint(0, n_valid - span_len)
                    
                    for offset in range(span_len):
                        if start_idx + offset < n_valid:
                            masked_indices_set.add(start_idx + offset)
                    
                    attempts += 1
                
                for idx in masked_indices_set:
                    pos = valid_positions[idx].item()
                    labels[i, pos] = input_ids[i, pos]
                    masked_input_ids[i, pos] = tokenizer.mask_id
                    
            else:
                perm = torch.randperm(n_valid, device=device)[:n_to_mask]
                positions_to_mask = valid_positions[perm]
                labels[i, positions_to_mask] = input_ids[i, positions_to_mask]
                masked_input_ids[i, positions_to_mask] = tokenizer.mask_id
            
    mask_counts = (masked_input_ids == tokenizer.mask_id).sum(dim=1).float()
    seq_lengths = (~is_special).sum(dim=1).float().clamp(min=1.0)
    timesteps = mask_counts / seq_lengths
    attn_mask = (masked_input_ids != tokenizer.pad_id).long()
    
    return masked_input_ids, labels, attn_mask, timesteps

# ============================================================================
# DATASET CLASS
# ============================================================================

class PI1MDataset(Dataset):
    """Dataset for polymer PSELFIES (unlabeled pretraining)."""
    
    def __init__(self, csv_path, tokenizer, max_length=128):
        print(f"Loading dataset from: {csv_path}")
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Look for the PSELFIES column specifically
        if 'pselfies' in self.data.columns:
            self.data = self.data.dropna(subset=['pselfies'])
            self.smiles = self.data['pselfies'].values
        elif 'pselfie' in self.data.columns:
            self.data = self.data.dropna(subset=['pselfie'])
            self.smiles = self.data['pselfie'].values
        elif 'p-SMILES' in self.data.columns:
            self.data = self.data.dropna(subset=['p-SMILES'])
            self.smiles = self.data['p-SMILES'].values
        else:
            self.smiles = self.data.iloc[:, 0].dropna().values
        
        print(f"Loaded {len(self.smiles)} PSELFIES structures")
        print(f"Sample PSELFIES: {self.smiles[0]}")
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = str(self.smiles[idx])
        encoded = self.tokenizer.encode(smiles, add_special_tokens=True)
        
        if len(encoded) < self.max_length:
            attention_mask = [1] * len(encoded) + [0] * (self.max_length - len(encoded))
            encoded = encoded + [self.tokenizer.pad_id] * (self.max_length - len(encoded))
        
        elif len(encoded) > self.max_length:
            # Truncate to max_length-1, then force EOS at the end
            encoded = encoded[:self.max_length - 1]
            encoded.append(self.tokenizer.eos_id)
            attention_mask = [1] * self.max_length  # All valid tokens (no PAD)
        
        else:
            # Exactly max_length — already fits perfectly
            attention_mask = [1] * self.max_length
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

class DiffusionCollator:
    """Data collator that applies random length exposure and span masking."""
    
    def __init__(self, tokenizer, random_length_prob=0.15):
        self.tokenizer = tokenizer
        self.random_length_prob = random_length_prob
        
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        batch_size = input_ids.size(0)
        
        # 1. Random length exposure (teaches model to stop early)
        for i in range(batch_size):
            if torch.rand(1).item() < self.random_length_prob:
                true_len = attention_mask[i].sum().item()
                if true_len > 3:
                    new_len = torch.randint(3, true_len + 1, (1,)).item()
                    input_ids[i, new_len - 1] = self.tokenizer.eos_id
                    input_ids[i, new_len:] = self.tokenizer.pad_id
                    attention_mask[i, new_len:] = 0
                    
        # 2. Generate random mask ratios (between 10% and 90%)
        mask_ratios = torch.rand(batch_size) * 0.80 + 0.10
        
        # 3. Apply optimized span masking
        masked_input_ids, labels, _, _ = apply_diffusion_masking(
            input_ids, mask_ratios, self.tokenizer, 'cpu', use_span_masking=True
        )
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'mask_ratios': mask_ratios,
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def save_checkpoint(model, optimizer, scheduler, history, global_step, epoch, output_dir, loss, tokenizer):
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'vocab_size': tokenizer.vocab_size,
        'tokenizer_vocab': tokenizer.vocab,
        'history': history,
    }, save_path / "model.pt")

    tokenizer.save(str(save_path / "tokenizer.pt"))
    
    return save_path

def create_interrupt_handler(model, optimizer, scheduler, history, output_dir, tokenizer):
    def save_on_interrupt(signum, frame):
        print("\n\n🛑 Interrupt detected! Saving checkpoint...")
        global_step = history['steps'][-1] if history['steps'] else 0
        epoch = len(set(history.get('epochs', [0])))
        loss = history['train_loss'][-1] if history['train_loss'] else 0
        
        interrupt_path = save_checkpoint(
            model, optimizer, scheduler, history, global_step, epoch, output_dir, loss, tokenizer
        )
        print(f"✓ Emergency checkpoint saved to: {interrupt_path}")
        sys.exit(0)
    return save_on_interrupt

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
if __name__ == "__main__":
    # Create both the scratch and home directory branches cleanly
    Path(OUTPUT_SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_HOME_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("LOADING DATA & BUILDING PSELFIES VOCABULARY")
    print("="*60)

    data = pd.read_csv(DATA_PATH)
    # Target the pselfie column
    if 'pselfies' in data.columns:
        smiles_col = 'pselfies'
    elif 'pselfie' in data.columns:
        smiles_col = 'pselfie'
    else:
        smiles_col = data.columns[0]
    smiles_list = data[smiles_col].dropna().values

    tokenizer = PSELFIESTokenizer(max_length=MAX_LENGTH)
    tokenizer.build_vocab(smiles_list, min_freq=2)
    vocab_size = tokenizer.vocab_size

    # ── VALIDATION UPGRADE: DATA SPLIT SEPARATION ──
    train_size = int(0.95 * len(smiles_list))
    train_smiles = smiles_list[:train_size]
    val_smiles = smiles_list[train_size:]

    train_dataset = PI1MDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    # Patch the internal arrays to reflect the split cleanly
    train_dataset.smiles = train_smiles
    
    val_dataset = PI1MDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    val_dataset.smiles = val_smiles

    print(f"Split data into {len(train_smiles)} train items and {len(val_smiles)} validation items")

    special_tokens = {
        'mask': tokenizer.mask_id, 'pad': tokenizer.pad_id,
        'bos': tokenizer.bos_id, 'eos': tokenizer.eos_id
    }
    
    # ── OLD collator instantiation ──────────────────────────────
    # collator = DiffusionCollator(
    #     mask_token_id=special_tokens['mask'], pad_token_id=special_tokens['pad'],
    #     bos_token_id=special_tokens['bos'], eos_token_id=special_tokens['eos']
    # )
    # ─────────────────────────────────────────────────────────────
    
    # ── OLD collator instantiation with eos_mask_prob ───────────
    # collator = DiffusionCollator(
    #     mask_token_id=special_tokens['mask'], pad_token_id=special_tokens['pad'],
    #     bos_token_id=special_tokens['bos'], eos_token_id=special_tokens['eos'],
    #     eos_mask_prob=0.8  # NEW: force EOS into mask 80% of the time
    # )
    # # ─────────────────────────────────────────────────────────────

    # ── NEW collator instantiation with eos_mask_prob ───────────
    collator = DiffusionCollator(
        tokenizer=tokenizer,
        random_length_prob=0.15
    )
    # ─────────────────────────────────────────────────────────────

    dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=4, persistent_workers=True, drop_last=True
    )
    
    # Missing Validation Engine Stream Loader
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collator, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=4, persistent_workers=True
    )

    print("\n" + "="*60)
    print("INITIALIZING DUAL-HEAD MODERNBERT MODEL")
    print("="*60)

    import torch._dynamo
    import transformers.utils.generic as hf_generic
    import transformers.utils.output_capturing as hf_capture

    # Fix the internal tracking graph break
    torch._dynamo.allow_in_graph(hf_generic.ContextManagers)

    # Inject 'torch' directly into HuggingFace modules so the compiler sees it
    hf_generic.torch = torch
    hf_capture.torch = torch

    model = ConditionalDiffusionLM(
        model_name=MODEL_NAME,
        vocab_size=vocab_size,
        use_property_conditioning=USE_PROPERTY_CONDITIONING,
        dropout=0.1
    ).to(device)

    if USE_COMPILE:
        try:
            print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode='default', fullgraph=False)
            with torch.no_grad():
                dummy_input = torch.randint(0, vocab_size, (BATCH_SIZE, 128), device=device)
                dummy_mask = torch.ones(BATCH_SIZE, 128, device=device)
                dummy_timestep = torch.rand(BATCH_SIZE, device=device)
                # Fix: Create a live dummy tensor to force the compiler to trace the conditioning layers
                dummy_prop = torch.rand(BATCH_SIZE, device=device)
                _ = model(input_ids=dummy_input, attention_mask=dummy_mask, timestep=dummy_timestep, prop=dummy_prop)
            print("✓ Model compiled and tested successfully!")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}")
            USE_COMPILE = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    # criterion = nn.CrossEntropyLoss(ignore_index=special_tokens['pad'])
    criterion_per_token = nn.CrossEntropyLoss(ignore_index=-100, reduction = 'none')
    
    # ── OLD eos_weight ──────────────────────────────────────────
    # eos_weight = 5.0
    # ─────────────────────────────────────────────────────────────
    
    # ── NEW eos_weight ──────────────────────────────────────────
    eos_weight = 10.0  # NEW: upweight EOS to combat ~20:1 frequency imbalance vs [C]f
    # ─────────────────────────────────────────────────────────────
    
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and device.type == 'cuda') else None

    history = {'train_loss': [], 'learning_rate': [], 'steps': [], 'epochs': [], 'val_loss': []}
    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0

    # ── NEW LOADING ENGINE: RESUME FROM CHECKPOINT ───────────────────
    if RESUME_FROM_STEP is not None:
        checkpoint_path = Path(OUTPUT_SCRATCH_DIR) / f"checkpoint-{RESUME_FROM_STEP}" / "model.pt"
        if checkpoint_path.exists():
            print(f"\n🔄 Resuming diffusion pretraining from step {RESUME_FROM_STEP}...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            global_step = checkpoint['step']
            start_epoch = checkpoint['epoch']
            history = checkpoint['history']
            print(f"✓ Successfully restored state dicts. Restarting at Epoch {start_epoch + 1}")
        else:
            print(f"⚠️ Checkpoint path '{checkpoint_path}' not found! Starting from scratch.")
    # ────────────────────────────────────────────────────────────────

    interrupt_handler = create_interrupt_handler(model, optimizer, scheduler, history, OUTPUT_SCRATCH_DIR, tokenizer)
    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)

    print("\n" + "="*60)
    print("STARTING PRETRAINING")
    print("="*60)

    model.train()

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}", total=len(dataloader))
        
        for batch_idx, batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            mask_ratios = batch['mask_ratios'].to(device, non_blocking=True)
            
            # 1. SYNTHETIC PROPERTY GENERATION: Generate random target values on the fly
            current_batch_size = input_ids.size(0)
            synthetic_properties = torch.rand(current_batch_size, device=device)
            
            # 2. CLASSIFIER-FREE GUIDANCE DROPOUT: Route 15% of inputs through an unconditional path
            cfg_dropout_mask = torch.rand(current_batch_size, device=device) < 0.15
            synthetic_properties[cfg_dropout_mask] = -1.0  # Sentinel indicator for [NULL] tracking
            
            with torch.amp.autocast(device_type=device.type, enabled=(USE_AMP and device.type == 'cuda')):
                # Unpack BOTH logits and values to anchor the critic network
                logits, values = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    timestep=mask_ratios,
                    prop=synthetic_properties  # Feed the active property vector
                )
                
                # A. Diffusion MLM Loss (Calculated strictly on masked targets)
                # OLD
                # diffusion_loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

                # CURRENT
                per_token_loss = criterion_per_token(logits.view(-1, vocab_size), labels.view(-1))

                weights = torch.ones_like(labels.view(-1), dtype=torch.float)
                weights[labels.view(-1) == tokenizer.eos_id] = eos_weight
                diffusion_loss = (per_token_loss * weights).mean()
                
                # B. Value Head Anchor Loss (Stabilizes critic baseline at 0.0 for real data)
                value_loss = nn.functional.mse_loss(values, torch.zeros_like(values))
                
                # Joint multi-task training loss calculation
                loss = (diffusion_loss + 0.1 * value_loss) / GRADIENT_ACCUMULATION_STEPS
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                global_step += 1

                # ── PLOTTING UPGRADE: RECORD HIGH-FREQUENCY METRICS ──
                if global_step % 10 == 0:
                    history['steps'].append(global_step)
                    history['train_loss'].append(loss.item() * GRADIENT_ACCUMULATION_STEPS)
                    # Track learning rate progression
                    history['learning_rate'].append(scheduler.get_last_lr()[0])
                
                pbar.set_postfix({
                    'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}", 
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                if global_step % SAVE_STEPS == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, history, global_step, epoch, OUTPUT_SCRATCH_DIR, loss.item() * GRADIENT_ACCUMULATION_STEPS, tokenizer
                    )

        # ── VALIDATION UPGRADE: EVALUATION HOOK PASS ──
        model.eval()
        val_loss = 0.0
        print(f"\n🧪 Running validation evaluations for Epoch {epoch+1}...")
        
        # ── NEW: EOS accuracy tracking counters ───────────────────
        total_eos_correct = 0
        total_eos_masked = 0
        # ─────────────────────────────────────────────────────────────
        
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc="Validation"):
                v_input_ids = val_batch['input_ids'].to(device, non_blocking=True)
                v_attn_mask = val_batch['attention_mask'].to(device, non_blocking=True)
                v_labels = val_batch['labels'].to(device, non_blocking=True)
                v_mask_ratios = val_batch['mask_ratios'].to(device, non_blocking=True)
                
                # Synthetic properties must match structural shape rules
                v_batch_size = v_input_ids.size(0)
                v_properties = torch.rand(v_batch_size, device=device)
                v_cfg_mask = torch.rand(v_batch_size, device=device) < 0.15
                v_properties[v_cfg_mask] = -1.0
                
                with torch.amp.autocast(device_type=device.type, enabled=(USE_AMP and device.type == 'cuda')):
                    v_logits, v_values = model(
                        input_ids=v_input_ids, attention_mask=v_attn_mask,
                        timestep=v_mask_ratios, prop=v_properties
                    )
                    # OLD
                    # v_diff_loss = criterion(v_logits.view(-1, vocab_size), v_labels.view(-1))
                    
                    # Current
                    v_per_token_loss = criterion_per_token(v_logits.view(-1, vocab_size), v_labels.view(-1))
                    v_weights = torch.ones_like(v_labels.view(-1), dtype=torch.float)
                    v_weights[v_labels.view(-1) == tokenizer.eos_id] = eos_weight
                    v_diff_loss = (v_per_token_loss * v_weights).mean()
                    
                    v_val_loss = nn.functional.mse_loss(v_values, torch.zeros_like(v_values))
                    
                    batch_loss = v_diff_loss + 0.1 * v_val_loss
                    
                    # ── NEW: track EOS accuracy on masked positions ───
                    v_labels_flat = v_labels.view(-1)
                    v_logits_flat = v_logits.view(-1, vocab_size)
                    eos_mask_flat = (v_labels_flat == tokenizer.eos_id)
                    if eos_mask_flat.any():
                        eos_preds = v_logits_flat[eos_mask_flat].argmax(-1)
                        total_eos_correct += (eos_preds == tokenizer.eos_id).sum().item()
                        total_eos_masked += eos_mask_flat.sum().item()
                    # ───────────────────────────────────────────────────
                
                val_loss += batch_loss.item()
                
        val_loss /= len(val_dataloader)
        epoch_loss /= len(dataloader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(OUTPUT_HOME_DIR) / "best_pretrain"
            best_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(best_path / "tokenizer.pt"))
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'tokenizer_vocab': tokenizer.vocab,
                'total_steps': global_step,
                'epochs': EPOCHS,
            }, best_path / "best_pretrain.pt")
            print(f"✓ New best model saved (val_loss={val_loss:.4f})")
        
        # Record everything cleanly to history arrays
        history['epochs'].append(epoch + 1)
        # Ensure your history dictionary has this key initialized at top if plotting!
        if 'val_loss' not in history:
            history['val_loss'] = []
        history['val_loss'].append(val_loss)
        
        # ── OLD epoch summary print ─────────────────────────────────
        # print(f"📊 Epoch {epoch+1} Results -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        # ─────────────────────────────────────────────────────────────
        
        # ── NEW epoch summary print with EOS accuracy ─────────────────
        print(f"📊 Epoch {epoch+1} Results -> Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}", end="")
        if total_eos_masked > 0:
            eos_acc = total_eos_correct / total_eos_masked
            print(f" | EOS Acc (masked): {eos_acc:.3f} ({total_eos_correct}/{total_eos_masked})")
        else:
            print(" | No EOS masked in validation")
        # ─────────────────────────────────────────────────────────────
        
        # Reset model state back to train mode for the next epoch iteration
        model.train()

    # Save final model
    print("\n" + "="*60)
    print("SAVING FINAL MODEL & TOKENIZER")
    print("="*60)

    final_path = Path(OUTPUT_HOME_DIR) / "best_pretrain"
    final_path.mkdir(parents=True, exist_ok=True)

    # ── CRITICAL EDIT 3: Explicitly saving the tokenizer file ─────────────────
    tokenizer.save(str(final_path / "tokenizer.pt"))
    # ───────────────────────────────────────────────────────────────────────────

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'tokenizer_vocab': tokenizer.vocab,
        'total_steps': global_step,
        'epochs': EPOCHS,
    }, final_path / "last_pretrain.pt")

    print(f"✓ Last epoch model saved to: {final_path / 'last_pretrain.pt'}")
    print(f"✓ Best model already saved to: {final_path / 'best_pretrain.pt'} (val_loss={best_val_loss:.4f})")
    print(f"✓ Tokenizer saved to: {final_path / 'tokenizer.pt'}")

    # ── PLOTTING UPGRADE: 3-PANEL CONFIGURATION DIAGNOSTICS CHANNELS ──
    try:
        print("\n📊 Generating diffusion pretraining diagnostic plots...")
        import matplotlib
        matplotlib.use('Agg') 
        import matplotlib.pyplot as plt

        plot_path = Path(PLOT_DIR)
        plot_path.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

        # Panel 1: Step-Level Diffusion MLM Optimization Trend
        ax1.plot(history['steps'], history['train_loss'], color='#1f77b4', alpha=0.3, label='Raw Batch Loss')
        if len(history['train_loss']) > 20:
            window_size = 20
            smoothed = pd.Series(history['train_loss']).rolling(window=window_size, min_periods=1).mean()
            ax1.plot(history['steps'], smoothed, color='#e377c2', linewidth=2, label='Smoothed Loss Trend')
        ax1.set_title('Masked Diffusion Pretraining Loss Profile', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Global Training Steps', fontsize=10)
        ax1.set_ylabel('Objective Value Loss (MLM + Critic Anchor)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # Panel 2: Epoch-Level Out-Of-Sample Validation Profile
        ax2.plot(history['epochs'], history['val_loss'], color='#2ca02c', marker='o', linewidth=2, label='Validation Loss')
        ax2.set_title('Validation Convergence Profile', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Completed Epochs', fontsize=10)
        ax2.set_ylabel('Total Evaluation Loss', fontsize=10)
        ax2.set_xticks(history['epochs'])
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        # Panel 3: Learning Rate Schedule Track Curve
        ax3.plot(history['steps'], history['learning_rate'], color='#bcbd22', linewidth=2, label='Cosine Schedule')
        ax3.set_title('Learning Rate Annealing Optimization Track', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Global Training Steps', fontsize=10)
        ax3.set_ylabel('Effective Learning Rate Value', fontsize=10)
        ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='upper right')

        plt.tight_layout()
        export_img_file = plot_path / 'diffusion_pretrain_metrics.png'
        plt.savefig(export_img_file, dpi=300)
        plt.close()
        print(f"✓ Diagnostic loss validation plot saved to: {export_img_file}")

        # Export structured tracking tables to disk
        export_csv_file = plot_path / 'diffusion_pretrain_metrics.csv'
        metrics_df = pd.DataFrame({
            "step": history['steps'],
            "loss": history['train_loss'],
            "lr": history['learning_rate']
        })
        metrics_df.to_csv(export_csv_file, index=False)
        print(f"✓ Structured diagnostics data table exported to: {export_csv_file}")

    except Exception as plotting_error:
        print(f"⚠ Diagnostic visualization routine skipped: {plotting_error}")
    # ────────────────────────────────────────────────────────────────

    print("\n🎉 PRETRAINING COMPLETE! 🎉\n")