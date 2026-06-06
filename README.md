# Conditional Discrete Diffusion Polymer Generation with Monte Carlo Tree Search (MCTS)

This repository contains the advanced optimization and sequence discovery framework for polymers, marrying a Conditional Discrete Diffusion Language Model (CDDLM) with an active lookahead Monte Carlo Tree Search (MCTS) exploration engine.

## Overview
Rather than utilizing standard token-by-token sequence estimation loops, this framework manages macromolecular construction as an iterative token-denoising problem over a complete sequence block, guided dynamically by MCTS lookahead expansions.

* Bidirectional Diffusion Backbone: Features a deep bidirectional transformer encoder engineered to predict, reconstruct, and refine structural sequences from completely or partially masked states.
* Confidence-Guided Tree Search: Integrates a customized lookahead Monte Carlo Tree Search optimization engine. The token mask confidence values produced by the diffusion transformer serve as the prior selection policy probabilities to guide branch choices.
* Latent Property Conditioning: Conditioned explicitly via high-dimensional continuous frequency projections that map numerical electronic band gap (E_g) targets into latent transformer weights during fine-tuning phases.
* Syntactic Safety: Governed entirely within a specialized tokenizer_pselfies.py engine, bounding the generative denoising process within strictly valid polymer-adapted SELFIES token rules.

---

## Pretrain Structure
(diffusion_generations.svg)

---

## Repository Structure

* model.py — Deep bidirectional conditional transformer configuration, including timestep and target property frequency projection embedding layers.
* tokenizer_pselfies.py — Specialized vocabulary matrices and regular-expression handling rules for polymer-adapted text variations (PSELFIES).
* pretraining.py — Baseline generative script executing token mask prediction sequences across large unconditioned configurations.
* pretrain.sh — High-performance computing shell script for executing unconditioned baseline training on cluster nodes.
* finetune_training.py — Core active discovery script managing tree expansion parameters, rollout updates, value tracking, and guided generation logs.
* mcts.sh — Batch engine shell script running target property fine-tuning backed by global confidence-guided MCTS action selections.
* __init__.py — Python package initialization file.
* .gitignore — Specifies intentionally untracked files to maintain a clean repository space.

---

## Getting Started

### 1. Pre-training Phase
To run unconditioned structural text grammar training to capture baseline chemical syntax across your cluster space allocation, execute:
bash pretrain.sh

### 2. MCTS Diffusion Fine-Tuning Phase
To execute target electronic property optimization driven by lookahead MCTS branch selections and diffusion token updates, run the primary batch engine script:
bash mcts.sh

---

## Research Attribution
This codebase is a component of ongoing graduate research at the Georgia Institute of Technology (School of Materials Science & Engineering).

Copyright & Licensing
© 2026 Vansh Suresh Yadav. All rights reserved.
This code is intended exclusively for private research evaluation. Copying, distributing, or modifying these files without explicit authorization is strictly prohibited.
