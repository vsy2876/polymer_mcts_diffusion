#!/usr/bin/env python
"""
POLYMER PSELFIES DIFFUSION MODEL - ALPHA ZERO FINE-TUNING SCRIPT
Replaces supervised data loading with MCTS self-play over a continuous Gaussian reward landscape.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import signal
import sys
import math
import random
from collections import deque
from pathlib import Path
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

# Import custom modules
from model import ConditionalDiffusionLM
from tokenizer_pselfies import PSELFIESTokenizer  # Upgraded to your PSELFIES script
import tempfile
import subprocess
import re
import selfies
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================================
# CONFIGURATION
# ============================================================================
PRETRAIN_CHECKPOINT = "./checkpoints/pretrain/best_pretrain/best_pretrain.pt"
PROPERTY_MIN = 0.5
PROPERTY_MAX = 10.0
OUTPUT_SCRATCH_DIR = '/storage/home/hcoda1/2/vyadav68/scratch/polymers/mcts_diffusion/checkpoints/mcts_diffusion'
OUTPUT_HOME_DIR    = './checkpoints/mcts_diffusion'

MAX_LENGTH = 128     # Adjusted to PSELFIES pretrain configuration
ITERATIONS = 5000    # Number of self-play training loops
N_SIMULATIONS = 40   # MCTS lookahead rollouts per masking step
C_PUCT = 1.5         # Exploration constant balancing policy vs value weights
TOP_K_TOKENS = 5     # Tree branching pruning factor (limits action space width)
PROPERTY_CONDITIONING = True   # Uses property conditioning
USE_COMPILE = True             # Toggle flag for fullgraph compilation tracking

BATCH_SIZE = 64  
REPLAY_BUFFER_MAX = 5000
MIN_BUFFER_SIZE = 256
LEARNING_RATE_BACKBONE = 2e-6  # Micro-LR to anchor core PSELFIES chemistry grammar
LEARNING_RATE_HEADS = 5e-5     # Faster learning rate for conditioning and heads
MAX_GRAD_NORM = 1.0
KL_BETA = 0.01
SIGMA_START = 0.5      # Wide reward landscape at start
SIGMA_END = 0.15       # Tighter target precision at end
SIGMA_ANNEAL_START = 2000   # Iteration to begin tightening
SIGMA_ANNEAL_END = 4000     # Iteration to reach final sigma

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# ── COMMENTED OUT OLD SUPERVISED DATASET & DATA LOADING COMPONENTS ──────────
# ============================================================================
"""
class PropertyDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256, property_col='Egc'):
        # Loaded static records...
        pass
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx): pass

class DiffusionCollator:
    def __init__(self, mask_token_id, pad_token_id, bos_token_id, eos_token_id):
        # Applied random masking arrays...
        pass
    def __call__(self, batch): pass
"""
# ============================================================================

# ============================================================================
# NEW ALPHA ZERO ADDITION: DISCRETE MASKED DIFFUSION MCTS GRAPH ENGINE
# ============================================================================
class MCTSNode:
    """Represents a state in the tree: a snapshot of the unmasking sequence."""
    def __init__(self, state_tensor, parent=None, prior_prob=0.0, chosen_pos=None, chosen_token=None):
        self.state = state_tensor.clone()  # (MAX_LENGTH,) array tracking tokens and [MASK]s
        self.parent = parent
        self.P = prior_prob                # Probability assigned by the policy head
        self.chosen_pos = chosen_pos       # Spatial index unmasked to reach this node
        self.chosen_token = chosen_token   # Token value assigned to that position
        self.children = {}                 # Maps (pos, token) -> MCTSNode
        self.N = 0                         # Visit count
        self.W = 0.0                       # Total accumulated reward
        self.Q = 0.0                       # Mean value score (W / N)

class MCTSDiffusionSearch:
    """Executes discrete lookahead searches by unmasking highly-confident tokens."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.special = {
            'mask': tokenizer.mask_id, 'bos': tokenizer.bos_id,
            'pad': tokenizer.pad_id, 'eos': tokenizer.eos_id
        }

        self.true_oracle_calls = 0 # Counter for actual xTB evaluations (for logging and analysis)

    def get_puct(self, parent, child):
        """Calculates Upper Confidence Bound score for node exploration selection."""
        u = C_PUCT * child.P * math.sqrt(parent.N) / (1 + child.N)
        return child.Q + u

    def select(self, node):
        """Descends the tree by selecting the child maximizing the PUCT score."""
        while node.children:
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = self.get_puct(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    def expand_and_evaluate(self, node, target_raw):
        mask_indices = (node.state == self.special['mask']).nonzero(as_tuple=True)[0]

        eval_batch_size = 64 if USE_COMPILE else 1
        input_ids = node.state.unsqueeze(0).repeat(eval_batch_size, 1).to(device)
        attn_mask = (input_ids != self.special['pad']).long()
        ratio_masked = len(mask_indices) / MAX_LENGTH
        
        t_tensor = torch.full((eval_batch_size,), ratio_masked, device=device)
        p_tensor = torch.full((eval_batch_size,), target_raw, device=device)

        with torch.no_grad():
            logits, values = self.model(input_ids, attn_mask, t_tensor, p_tensor)
            logits_filtered = logits[0].clone()
            predicted_value = values[0].item()
            
        # ✅ FIX: If there are no masks left, this is a terminal state in the tree search.
        # Return the neural network's own Value Head prediction proxy instantly instead of running xTB!
        if len(mask_indices) == 0:
            return predicted_value
            
        for sid in [self.special['bos'], self.special['mask'], self.tokenizer.unk_id]:
            logits_filtered[:, sid] = -float('inf')
            
        probs = F.softmax(logits_filtered, dim=-1)

        # GLOBAL CONFIDENCE-GUIDED ACTION SELECTION
        mask_probs = probs[mask_indices.to(device)]
        entropy = -torch.sum(mask_probs * torch.log(mask_probs + 1e-9), dim=-1)
        top2_probs, top2_indices = torch.topk(mask_probs, 2, dim=-1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        scores = top2_probs[:, 0] * torch.exp(-entropy) * torch.sigmoid(margin)
        
        k = min(TOP_K_TOKENS, len(mask_indices))
        topk_scores, topk_rank_indices = torch.topk(scores, k)
        
        selected_positions = mask_indices[topk_rank_indices.cpu()]
        selected_tokens = top2_indices[topk_rank_indices, 0]
        
        topk_probs = top2_probs[topk_rank_indices, 0]
        topk_probs = topk_probs / topk_probs.sum()
        
        if node.parent is None:
            noise = torch.distributions.Dirichlet(torch.full((k,), 0.3)).sample().to(device)
            topk_probs = 0.75 * topk_probs + 0.25 * noise
            
        for p, pos, tok in zip(topk_probs.cpu().tolist(), selected_positions.cpu().tolist(), selected_tokens.cpu().tolist()):
            next_state = node.state.clone()
            next_state[pos] = tok
            child = MCTSNode(next_state, parent=node, prior_prob=p, chosen_pos=pos, chosen_token=tok)
            node.children[(pos, tok)] = child

        return predicted_value

    def _execute_composite_calculation(self, tmpdir, smiles_string, prefix):
        """Helper to run GFN2 geometry optimization followed by a g-xTB single-point gap check."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import subprocess
        import os
        import re

        try:
            # Safely check parsing
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                return -1.0
            
            mol = Chem.AddHs(mol)
            # Switch embedding failure from None to -1.0
            if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
                return -1.0
            
            # INSULATE FORCE FIELD OPTIMIZATION AGAINST EXOTIC ATOMS
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                return -1.0
            
            xyz_path = os.path.join(tmpdir, f"{prefix}_input.xyz")
            Chem.MolToXYZFile(mol, xyz_path)

            # STAGE 1: Geometry Optimization via GFN2-xTB
            try:
                result_opt = subprocess.run(
                    ["xtb", f"{prefix}_input.xyz", "--gfn", "2", "--opt"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                return -1.0
            
            if result_opt.returncode != 0:
                return -1.0

            opt_xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
            dest_opt_xyz = os.path.join(tmpdir, f"{prefix}_opt.xyz")
            if os.path.exists(opt_xyz_path):
                os.rename(opt_xyz_path, dest_opt_xyz)
            else:
                return -1.0

            # STAGE 2: High-Precision Single-Point via g-xTB
            try:
                result_gxtb = subprocess.run(
                    ["xtb", f"{prefix}_opt.xyz", "--gxtb"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                return -1.0

            if result_gxtb.returncode != 0:
                return -1.0

            output_text = result_gxtb.stdout
            match = re.search(r"(?:HOMO-LUMO\s+gap|HL-Gap)\s*[:\/\s\w]*\s*([-+]?\d*\.\d+|\d+)", output_text, re.IGNORECASE)
            if match:
                return float(match.group(1))
                
            for line in output_text.splitlines():
                if "gap" in line.lower() or "hl-gap" in line.lower():
                    parts = line.split()
                    for part in parts:
                        try:
                            val = float(part)
                            if 0.0 < val < 30.0:
                                return val
                        except ValueError:
                            continue
            return -1.0
            
        except Exception:
            # Global fallback catch-all to prevent any unexpected execution failures
            return -1.0

    def evaluate_terminal(self, node, target_raw, sigma):
        """Calculates infinite-chain polymer band gap using an oligomeric 1/N scaling law extrapolation."""
        import csv
        import os

        log_path = f"{OUTPUT_HOME_DIR}/diffusion_sequence_log.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_exists = os.path.isfile(log_path)

        try:
            raw_string = self.tokenizer.decode(node.state.tolist(), skip_special_tokens=False)
            pselfies_clean = raw_string.replace("<BOS>","").replace("<EOS>","").replace("<PAD>","").replace("[MASK]","").strip()
            
            smiles = selfies.decoder(pselfies_clean)
            if not smiles:
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                    writer.writerow([pselfies_clean, "", "", False, -1.0])
                return -1.0
            
            # # Hardened multi-pass structural re-contextualization engine
            # core_check = smiles.replace("[At]", "").replace("At", "").replace("[*]", "").replace("*", "")
            # prev = None
            # while prev != core_check:
            #     prev = core_check
            #     core_check = re.sub(r'\([=#/\\]?\)', '', core_check)
            #     core_check = re.sub(r'[=#/\\]+\)', ')', core_check)
            #     core_check = re.sub(r'\([^a-zA-Z0-9]*\)', '', core_check)
            #     core_check = re.sub(r'[=#/\\]+$', '', core_check.strip())
            #     core_check = re.sub(r'^[=#/\\]+', '', core_check.strip())
            
            # if "=" not in core_check:
            #     core_check = core_check.replace("/", "").replace("\\", "")
            # else:
            #     core_check = re.sub(r'(?<=[\(\)])[/\\]+|[/\\]+(?=[\(\)])', '', core_check)
            #     core_check = re.sub(r'[/\\]{2,}', '', core_check)

            smiles_clean = smiles.replace("[At]", "").replace("At", "").strip()     
            psmiles_raw = pselfies_clean
            smiles_raw = smiles
            # smiles_clean = core_check.strip()

            smiles = smiles.replace("[At]", "").replace("At", "")
            smiles_clean = smiles.strip()

            from rdkit import Chem
            mol_check = Chem.MolFromSmiles(f"[H]{smiles_clean}[H]") if smiles_clean else None
            
            # CRITICAL VERIFICATION: Validate structure rules BEFORE printing to prevent double rows
            if not smiles_clean or mol_check is None:
                with open(log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                    writer.writerow([psmiles_raw, smiles_raw, smiles_clean, False, -1.0])
                return -1.0

            # Write the initial tracking state cleanly on a single row pass
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["pselfies_raw", "smiles_before_regex", "smiles_after_regex", "rdkit_valid", "reward"])
                writer.writerow([psmiles_raw, smiles_raw, smiles_clean, True, "pending"])

            self.true_oracle_calls += 2  
            
            monomer_smiles = f"[H]{smiles_clean}[H]"
            trimer_smiles  = f"[H]{smiles_clean}{smiles_clean}{smiles_clean}[H]"
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                gap_n1 = self._execute_composite_calculation(tmp_dir, monomer_smiles, "monomer")
                if gap_n1 is None or gap_n1 < 0.0 or gap_n1 > 30.0:
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0
                    
                gap_n3 = self._execute_composite_calculation(tmp_dir, trimer_smiles, "trimer")
                if gap_n3 is None or gap_n3 < 0.0 or gap_n3 > 30.0:
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0
                    
                infinite_gap = (3.0 * gap_n3 - gap_n1) / 2.0
                if not (0.0 < infinite_gap < 30.0):
                    with open(log_path, 'a', newline='') as f:
                        csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, -1.0])
                    return -1.0
                
                reward = math.exp(-((infinite_gap - target_raw) ** 2) / (2 * sigma ** 2))
                final_score = (reward * 2.0) - 1.0
                
                with open(log_path, 'a', newline='') as f:
                    csv.writer(f).writerow([psmiles_raw, smiles_raw, smiles_clean, True, final_score])
                return final_score
                
        except Exception:
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([pselfies_clean, "", "", False, -1.0])
            return -1.0

    def backpropagate(self, node, value):
        """Updates visit tallies and running average value approximations up the branch."""
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    # 1. Inside MCTSDiffusionSearch.run_search:
    def run_search(self, target_norm, sigma):
        root_tensor = torch.full((MAX_LENGTH,), self.special['mask'], dtype=torch.long)
        root_tensor[0] = self.special['bos']
        root = MCTSNode(root_tensor)

        # Convert to physical scale right at the start of the episode
        target_raw = target_norm * (PROPERTY_MAX - PROPERTY_MIN) + PROPERTY_MIN
        trajectory = []
        
        for step in range(MAX_LENGTH - 1):
            mask_indices = (root.state == self.special['mask']).nonzero(as_tuple=True)[0]
            if len(mask_indices) == 0:
                break

            for _ in range(N_SIMULATIONS):
                leaf = self.select(root)
                # Pass target_raw down instead of target_norm
                value = self.expand_and_evaluate(leaf, target_raw) 
                self.backpropagate(leaf, value)

            if not root.children:
                break

            actions = list(root.children.keys())
            counts = torch.tensor([root.children[a].N for a in actions], dtype=torch.float32)
            probs = counts / counts.sum()

            chosen_idx = torch.multinomial(probs, 1).item()
            chosen_action = actions[chosen_idx]

            # Store target_raw in the buffer memory slot
            trajectory.append((root.state.clone(), chosen_action[0], chosen_action[1], target_raw))

            # ... (MCTS selects chosen_action and appends to trajectory) ...
            root = root.children[chosen_action]
            root.parent = None

            # NEW: DYNAMIC EARLY EXIT FOR SYSTEM VELOCITY
            if chosen_action[1] == self.special['eos']:
                eos_position = chosen_action[0]

                all_masks = (root.state == self.special['mask']).nonzero(as_tuple=True)[0]
                
                # Isolate only the masking slots located strictly to the right of the EOS marker
                trailing_masks = [pos for pos in all_masks.cpu().tolist() if pos > eos_position]
                
                # Instantly map trailing masks to pad states in memory
                for pos in trailing_masks:
                    trajectory.append((root.state.clone(), pos, self.special['pad'], target_raw))
                    root.state[pos] = self.special['pad']

        final_reward = self.evaluate_terminal(root, target_raw, sigma)
        
        # Apply an exponential temporal decay scale over macro unmasking operations
        updated_trajectory = []
        T = len(trajectory)
        gamma = 0.99
        
        for t, (state, pos, tok, tgt) in enumerate(trajectory):
            discounted_reward = (gamma ** (T - 1 - t)) * final_reward
            updated_trajectory.append((state, pos, tok, tgt, discounted_reward, final_reward))
            
        return updated_trajectory
# ============================================================================

# ============================================================================
# INITIALIZATION & MODEL SETUP
# ============================================================================

def seed_diffusion_replay_buffer(csv_path, tokenizer, replay_buffer, target_samples=1500):
    """Parses Khazana database polymer targets and populates the replay buffer

    with simulated intermediate masked states paired with perfect rewards.
    """
    print(f"\n📥 Pre-seeding replay buffer with training view variations from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"⚠️ Target dataset path '{csv_path}' not found. Skipping buffer pre-seed loop.")
        return

    df = pd.read_csv(csv_path)
    # Assume columns are named 'pselfies' and 'Egc' based on Khazana configuration
    pselfies_col = 'pselfies' if 'pselfies' in df.columns else df.columns[0]
    egc_col = 'Egc' if 'Egc' in df.columns else df.columns[1]

    sampled_records = df.sample(n=min(target_samples, len(df)), random_state=42).reset_index(drop=True)
    
    for _, row in sampled_records.iterrows():
        pselfies_str = str(row[pselfies_col])
        true_gap = float(row[egc_col])
        
        # Encode string to token sequence
        tokens = tokenizer.encode(pselfies_str, add_special_tokens=True)
        if len(tokens) > MAX_LENGTH:
            continue
            
        padded_tokens = tokens + [tokenizer.pad_id] * (MAX_LENGTH - len(tokens))
        full_sequence_tensor = torch.tensor(padded_tokens, dtype=torch.long)
        
        # Identify valid chemical positions that are allowed to be masked/unmasked
        valid_positions = [
            i for i, tok in enumerate(padded_tokens)
            if tok not in [tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id]
        ]
        
        if not valid_positions:
            continue
            
        # Simulate 3 unique masking views per compound to give the value head diverse structural context
        for _ in range(3):
            chosen_pos = random.choice(valid_positions)
            chosen_tok = padded_tokens[chosen_pos]
            
            state_tensor = full_sequence_tensor.clone()
            state_tensor[chosen_pos] = tokenizer.mask_id
            
            # Mask a random fraction of other tokens to simulate an active MCTS search node snapshot
            mask_ratio = random.uniform(0.1, 0.6)
            num_extra_masks = int(len(valid_positions) * mask_ratio)
            extra_positions = random.sample(valid_positions, min(num_extra_masks, len(valid_positions)))
            
            for pos in extra_positions:
                state_tensor[pos] = tokenizer.mask_id
                
            # Double-check that our targeted item to guess remains explicitly masked
            state_tensor[chosen_pos] = tokenizer.mask_id
            
            # Since these are real structures, they represent ideal ground-truth physical targets
            target_raw = random.uniform(PROPERTY_MIN, PROPERTY_MAX)

            raw_reward = math.exp(-((true_gap - target_raw) ** 2) / (2 * 0.5 ** 2))
            final_reward = (raw_reward * 2.0) - 1.0 
            
            replay_buffer.append((state_tensor, chosen_pos, chosen_tok, target_raw, final_reward, final_reward))
            
    print(f"✓ Initialization Complete. Replay Buffer seeded with {len(replay_buffer)} entries.")

def get_current_sigma(iteration):
    """Linearly anneals sigma from SIGMA_START to SIGMA_END between anneal iterations."""
    if iteration < SIGMA_ANNEAL_START:
        return SIGMA_START
    if iteration >= SIGMA_ANNEAL_END:
        return SIGMA_END
    progress = (iteration - SIGMA_ANNEAL_START) / (SIGMA_ANNEAL_END - SIGMA_ANNEAL_START)
    return SIGMA_START + progress * (SIGMA_END - SIGMA_START)

if __name__ == "__main__":
    Path(OUTPUT_SCRATCH_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_HOME_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary from pretraining checkpoint parameters
    tokenizer = PSELFIESTokenizer.load(os.path.dirname(PRETRAIN_CHECKPOINT) + "/tokenizer.pt")
    
    # Setup our physics oracle parameters
    # (Surrogate model removed in favor of direct GFN2-xTB calculations)
    
    # Initialize conditioning model topology
    model = ConditionalDiffusionLM(vocab_size=tokenizer.vocab_size, use_property_conditioning=PROPERTY_CONDITIONING).to(device)
    pre_ckpt = torch.load(PRETRAIN_CHECKPOINT, map_location=device)

    state_dict = pre_ckpt["model_state_dict"]
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    
    backbone_frozen_params = []
    backbone_trainable_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "output_head" in name or "value_head" in name or "property" in name:
            # Task heads and conditioning blocks are always fully trainable
            param.requires_grad = True
            head_params.append(param)
        elif "layers.11" in name or "final_layer_norm" in name:
            # Unfreeze the final transformer block (Block 12) + layer norms to learn structural re-contextualization
            param.requires_grad = True
            backbone_trainable_params.append(param)
        else:
            # Hard freeze the base grammar backbone layers (Layers 1-11)
            param.requires_grad = False
            backbone_frozen_params.append(param)
            
    # Setup our differentiated optimizer group allocations
    optimizer = torch.optim.AdamW([
        {"params": backbone_trainable_params, "lr": LEARNING_RATE_BACKBONE}, # Higher learning rate sandbox for layer 12
        {"params": head_params, "lr": LEARNING_RATE_HEADS}
    ], weight_decay=0.01)
    
    # ── COMPILER ENVIRONMENT PATCHES ────────────────────────────────
    import torch._dynamo
    import transformers.utils.generic as hf_generic
    import transformers.utils.output_capturing as hf_capture

    # Direct Graph block permission injection
    torch._dynamo.allow_in_graph(hf_generic.ContextManagers)

    # Bind explicit global instances across environment scope drops
    hf_generic.torch = torch
    hf_capture.torch = torch
    # ────────────────────────────────────────────────────────────────

    # After loading model weights, before compile — create frozen reference
    ref_model = ConditionalDiffusionLM(
        vocab_size=tokenizer.vocab_size, 
        use_property_conditioning=PROPERTY_CONDITIONING
    ).to(device)
    ref_model.load_state_dict(state_dict)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # ── COMPILATION ROUTINE ─────────────────────────────────────────
    if USE_COMPILE:
        try:
            print("\nCompiling model with torch.compile()...")
            model = torch.compile(model, mode='default', fullgraph=False)
            
            with torch.no_grad():
                # Warm up using batch size 64 to perfectly align with search and optimization states
                dummy_states = torch.randint(0, tokenizer.vocab_size, (64, MAX_LENGTH), device=device)
                dummy_mask = (dummy_states != tokenizer.pad_id).long()
                dummy_timesteps = torch.rand(64, device=device)
                dummy_targets = torch.rand(64, device=device)
                
                _ = model(dummy_states, dummy_mask, dummy_timesteps, dummy_targets)
            print("✓ Model compiled and tested successfully with a 100% complete graph!")
        except Exception as e:
            import traceback
            print(f"⚠ Could not compile models: {e}")
            print("Full traceback:")
            traceback.print_exc()
            USE_COMPILE = False
    # ────────────────────────────────────────────────────────────────
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    search_engine = MCTSDiffusionSearch(model, tokenizer)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_MAX)

    # Point this to the physical path of your Khazana CSV file on the scratch directory
    seed_diffusion_replay_buffer("/storage/home/hcoda1/2/vyadav68/scratch/polymers/mcts_diffusion/Egc_pselfies.csv", tokenizer, replay_buffer)

    # ── NEW ALPHA ZERO ADDITION: SELF-PLAY TRAINING LOOP ────────────────────
    print("🚀 Starting AlphaZero Masked Discrete Diffusion Finetuning Loop...")
    pbar = tqdm(range(ITERATIONS))
    
    for iteration in pbar:
        # 1. Self-Play Episode Generation
        model.eval()
        target_norm = random.uniform(0.0, 1.0) # Uniformly sample property goals across the 0-10 eV distribution
        current_sigma = get_current_sigma(iteration)
        episode_data = search_engine.run_search(target_norm, current_sigma)
        replay_buffer.extend(episode_data)

        # Wait until the memory buffer has collected sufficient self-play transitions
        if len(replay_buffer) < MIN_BUFFER_SIZE:
            pbar.set_postfix({"Buffer Fill": f"{len(replay_buffer)}/{MIN_BUFFER_SIZE}"})
            continue

        # 2. Optimization Update Step over sampled batch
        batch = random.sample(replay_buffer, BATCH_SIZE)
        
        states = torch.stack([item[0] for item in batch]).to(device)
        positions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
        tokens = torch.tensor([item[2] for item in batch], dtype=torch.long, device=device)
        targets = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=device)
        rewards = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
        value_targets = torch.tensor([item[5] for item in batch], dtype=torch.float32, device=device)  # flat, for value head

        # Dynamically evaluate exact mask timestep ratio configurations per entry
        mask_counts = (states == tokenizer.mask_id).sum(dim=1).float()
        timesteps = mask_counts / MAX_LENGTH

        model.train()
        optimizer.zero_grad()
        
        with torch.no_grad():
            ref_logits, _ = ref_model(states, (states != tokenizer.pad_id).long(), timesteps, targets)

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits, values = model(states, (states != tokenizer.pad_id).long(), timesteps, targets)
            
            # POLICY HEAD LOSS
            target_logits = logits[torch.arange(len(batch), device=device), positions]
            per_sample_loss = F.cross_entropy(target_logits, tokens, reduction='none')
            
            # CORRECTED: Combine action error directly with the MCTS physics rewards
            # (No artificial EOS scaling here; let the xTB oracle completely guide the policy path!)
            policy_gradient_loss = per_sample_loss * rewards
            policy_loss = policy_gradient_loss.mean()
            
            # VALUE HEAD LOSS - Explicitly force float32 promotion here:
            extracted_values = values.view(-1).float()
            raw_value_loss = F.mse_loss(extracted_values, value_targets.view(-1), reduction='none')
            value_loss = raw_value_loss.mean()

            kl_loss = F.kl_div(
                F.log_softmax(logits.float(), dim=-1),
                F.softmax(ref_logits.float(), dim=-1),
                reduction='batchmean'
            )

            # Joint objective loss calculation
            total_loss = policy_loss + value_loss + KL_BETA * kl_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if (iteration + 1) % 10 == 0:
            avg_reward = sum(
                item[4] for item in list(replay_buffer)[-100:]
            ) / min(100, len(replay_buffer))
            
            print(
                f"Iter {iteration+1}/{ITERATIONS} | "
                f"sigma={current_sigma:.4f} "
                f"loss={total_loss.item():.4f} "
                f"policy={policy_loss.item():.4f} "
                f"value={value_loss.item():.4f} "         # should fall over time
                f"avg_reward(last100)={avg_reward:.3f} "  # key signal: is xTB returning real values?
                f"reward_mean={rewards.mean().item():.3f} "   # this batch's mean reward
                f"reward_std={rewards.std().item():.3f} "     # low std = value head collapsing to -1
                f"timestep_mean={timesteps.mean().item():.3f} " # masking ratio being seen
                f"buffer={len(replay_buffer)} "
                f"xtb_calls={search_engine.true_oracle_calls}"
            )

        # Save model parameters periodicallys
        if (iteration + 1) % 500 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_vocab": tokenizer.vocab,
                "vocab_size": tokenizer.vocab_size,
                "property_min": PROPERTY_MIN,
                "property_max": PROPERTY_MAX,
            }, f"{OUTPUT_SCRATCH_DIR}/mctd_model_iter_{iteration+1}.pt")

        if (iteration + 1) % 10 == 0:
            # 2. Append and export sample efficiency curves for plotting
            log_path = f"{OUTPUT_HOME_DIR}/sample_efficiency_metrics.csv"
            new_row = pd.DataFrame([{
                "iteration": iteration + 1,
                "true_oracle_calls": search_engine.true_oracle_calls,
                "loss": total_loss.item()
            }])
            if os.path.exists(log_path):
                new_row.to_csv(log_path, mode='a', header=False, index=False)
            else:
                new_row.to_csv(log_path, mode='w', header=True, index=False)

        # Final Iteration Production Save
        if (iteration + 1) == ITERATIONS:
            print(f"🎉 Training complete! Archiving final model state and metadata in home...")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_vocab":  tokenizer.vocab,
                "vocab_size":       tokenizer.vocab_size,
                "property_min":     PROPERTY_MIN,
                "property_max":     PROPERTY_MAX,
                "iteration":        iteration + 1,
            }, f"{OUTPUT_HOME_DIR}/final_mctd_diffusion_model.pt")
            
            # Explicitly save a clean copy of the tokenizer right next to it
            tokenizer.save(f"{OUTPUT_HOME_DIR}/tokenizer.pt")
    # ────────────────────────────────────────────────────────────────────