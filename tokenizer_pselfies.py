"""PSELFIES Tokenizer — bracket-grouped tokenization for polymer SELFIES."""
import re
import torch
from typing import List, Dict


class PSELFIESTokenizer:
    """Tokenizer that treats every [...] bracket as one token."""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        # Special tokens (fixed IDs)
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.mask_token = "[MASK]"
        self.unk_token = "<UNK>"

        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.mask_id = 3
        self.unk_id = 4

        # Register special tokens
        self.vocab[self.pad_token] = self.pad_id
        self.vocab[self.bos_token] = self.bos_id
        self.vocab[self.eos_token] = self.eos_id
        self.vocab[self.mask_token] = self.mask_id
        self.vocab[self.unk_token] = self.unk_id

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)  # Initialize here (5 for special tokens)

    def build_vocab(self, pselfies_list: List[str], min_freq: int = 1):
        """Build vocabulary from list of PSELFIES strings."""
        token_counts = {}
        pattern = re.compile(r"\[[^\]]+\]")

        for pselfies in pselfies_list:
            tokens = pattern.findall(pselfies)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

        # Add tokens with sufficient frequency
        next_id = max(self.vocab.values()) + 1
        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1

        self.vocab_size = len(self.vocab)
        print(f"Vocab size: {self.vocab_size}")
        return self

    def encode(self, pselfies: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a PSELFIES string to token IDs."""
        pattern = re.compile(r"\[[^\]]+\]")
        tokens = pattern.findall(pselfies)

        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)

        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)

        # Truncate if needed
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to PSELFIES string with automatic EOS termination."""
        tokens = []
        for idx in ids:
            token = self.inverse_vocab.get(idx, self.unk_token)
            
            # 1. Catch the model's EOS signal and stop decoding immediately
            if token == self.eos_token:
                break
                
            # 2. Silently skip formatting placeholders 
            if skip_special_tokens and token in [self.pad_token, self.bos_token]:
                continue
                
            tokens.append(token)
        return "".join(tokens)

    def __call__(self, pselfies_list: List[str], return_tensors: str = "pt") -> dict:
        """Batch encode with padding."""
        if isinstance(pselfies_list, str):
            pselfies_list = [pselfies_list]

        batch_ids = []
        for pselfies in pselfies_list:
            ids = self.encode(pselfies)
            # Right-pad to max_length
            if len(ids) < self.max_length:
                ids = ids + [self.pad_id] * (self.max_length - len(ids))
            batch_ids.append(ids)

        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}

    def save(self, path: str):
        """Save tokenizer to disk."""
        torch.save({
            "vocab": self.vocab,
            "max_length": self.max_length,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "mask_id": self.mask_id,
            "unk_id": self.unk_id,
        }, path)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load tokenizer from disk."""
        data = torch.load(path, weights_only=False)
        tokenizer = cls(max_length=data["max_length"])
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.pad_id = data["pad_id"]
        tokenizer.bos_id = data["bos_id"]
        tokenizer.eos_id = data["eos_id"]
        tokenizer.mask_id = data["mask_id"]
        tokenizer.unk_id = data["unk_id"]
        tokenizer.vocab_size = len(tokenizer.vocab)
        return tokenizer


if __name__ == "__main__":
    import pandas as pd
    import argparse

    # ── Config ──────────────────────────────────────────────────────
    class args:
        csv        = "/path/to/PI1M.csv"
        out        = "checkpoints/pretrain/tokenizer.pt"
        max_length = 128
        min_freq   = 1
    # ────────────────────────────────────────────────────────────────

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["pselfie"])

    tokenizer = PSELFIESTokenizer(max_length=128)
    tokenizer.build_vocab(df["pselfie"].tolist(), min_freq=1)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer.save(args.out)
