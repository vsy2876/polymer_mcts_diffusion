# Conditional Discrete Diffusion Polymer Generation with MCTS

This repository contains an advanced optimization and sequence discovery framework for polymers, marrying a **Conditional Discrete Diffusion Language Model** with a **Monte Carlo Tree Search (MCTS)** exploration engine.

## Architecture & Methodology
Rather than utilizing token-by-token sequence estimation loops, this framework manages molecule construction as an iterative token-denoising problem over a complete token space block, guided by MCTS sequence expansions.

* **Generative Backbone:** Bidirectional Diffusion Transformer Encoder engineered to predict and reconstruct structural sequences from completely or partially masked states.
* **Search Strategy:** Integrated look-ahead Monte Carlo Tree Search optimization engine. The token mask confidence values produced by the diffusion transformer serve as the prior selection policy probabilities to guide branch choices.
* **Property Target Tuning:** Conditioned explicitly via high-dimensional frequency arrays mapping numerical band gap ($E_{gc}$) attributes into latent transformer weights during fine-tuning phases.
* **Vocabulary Safety:** Governed completely within a specialized localized `tokenizer_pselfies.py` engine, bounding the generative process within correct polymer-adapted SELFIES token rules.

---

## Repository Structure

* `model.py` — Deep bidirectional conditional transformer layout including timestep and target frequency projection embeddings.
* `tokenizer_pselfies.py` — Specialized vocabulary matrices and regex expression handling rules for polymer-adapted text variations.
* `pretraining.py` / `pretrain.sh` — Baseline generative script executing token mask prediction sequences across large unconditioned configurations.
* `finetune_training.py` / `mcts.sh` — Core active discovery script managing the tree expansion parameters, rollout updates, value tracking, and guided generation logs.

---

## Getting Started

### 1. Pre-training Configuration
To run unconditioned structural text grammar training on your cluster space allocation:
`bash pretrain.sh`

### 2. MCTS Diffusion Generation Fine-Tuning
To run target property fine-tuning backed by global confidence-guided MCTS action selections:
`bash mcts.sh`

---

## Research Attribution
This codebase is a component of ongoing graduate research at the Georgia Institute of Technology (School of Materials Science & Engineering).

**Copyright & Licensing** © 2026 Vansh Suresh Yadav. All rights reserved.  
This code is intended exclusively for private research evaluation. Copying, distributing, or modifying these files without explicit authorization is strictly prohibited.
