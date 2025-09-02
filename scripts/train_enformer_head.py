#!/usr/bin/env python3
"""
train_enformer_head.py (PyTorch version)
---------------------------------------
• Loads a frozen Enformer trunk from HuggingFace (pytorch weights).
• Streams 196 608-bp windows (stride 50 kb) from BAM files in ~/tcgaData/.
• Extracts the CLS-token embedding (dim=1536) per window and feeds it to a
  lightweight multitask head (tumour-stage classification + IFN-γ regression).

Install deps (example):
    python -m venv env && source env/bin/activate
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    pip install enformer-pytorch pysam pyfaidx pandas tqdm scikit-learn

A CSV called labels.csv must exist with columns:
    sample_id,tumor_stage,ifng

This script is compute-heavy (Enformer forward pass) so adjust batch-size /
window stride as needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

try:
    from enformer_pytorch import Enformer
except ImportError as e:
    sys.exit("❌ enformer-pytorch not found. `pip install enformer-pytorch`.")

# ---------------------------------------------------------------------
WINDOW_SIZE = 196_608
STRIDE = 50_000
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def one_hot_encode(seq: str) -> torch.Tensor:
    arr = torch.zeros((len(seq), 4), dtype=torch.float32)
    for i, b in enumerate(seq.upper()):
        idx = NUC_TO_IDX.get(b)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def windows_for_contig(length: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i + WINDOW_SIZE <= length:
        out.append((i, i + WINDOW_SIZE))
        i += STRIDE
    return out


class BamWindowDataset(Dataset):
    """Yield (one-hot seq, stage, ifng) for every window in every BAM."""

    def __init__(self, bam_files: List[Path], fasta: Path, labels: pd.DataFrame):
        self.bam_files = bam_files
        self.ref = pysam.FastaFile(str(fasta))
        self.labels = labels.set_index("sample_id")
        self.index: List[Tuple[int, str, int, int]] = []  # (file_idx, chr, start, end)
        for fi, bam in enumerate(bam_files):
            with pysam.AlignmentFile(bam, "rb") as b:
                for chr_ in b.references:
                    length = b.get_reference_length(chr_)
                    for start, end in windows_for_contig(length):
                        self.index.append((fi, chr_, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, chr_, start, end = self.index[idx]
        bam_path = self.bam_files[fi]
        seq = self.ref.fetch(chr_, start, end)
        x = one_hot_encode(seq)  # (L,4)
        sample_id = bam_path.stem
        row = self.labels.loc[sample_id]
        return x, torch.tensor(row.tumor_stage, dtype=torch.long), torch.tensor(row.ifng, dtype=torch.float32)


def collate_fn(batch):
    xs, stages, ifngs = zip(*batch)
    xs = torch.stack(xs)  # (B, L, 4)
    return xs, torch.stack(stages), torch.stack(ifngs)


class Head(nn.Module):
    def __init__(self, in_dim: int = 1536):
        super().__init__()
        self.stage = nn.Linear(in_dim, 4)
        self.ifng = nn.Linear(in_dim, 1)

    def forward(self, x):  # x (B, D)
        return self.stage(x), self.ifng(x).squeeze(1)


def main():
    root = Path.home() / "tcgaData"
    bam_files = sorted(root.glob("*.bam"))
    if not bam_files:
        sys.exit(f"No BAM files in {root}")

    labels = pd.read_csv("labels.csv")

    ds = BamWindowDataset(bam_files, Path.home() / "reference/GRCh38.fa", labels)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    enformer = Enformer.from_pretrained("EleutherAI/enformer-pytorch").eval().to(DEVICE)
    for p in enformer.parameters():
        p.requires_grad = False

    head = Head().to(DEVICE)

    optim = torch.optim.Adam(head.parameters(), lr=3e-4)
    cls_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()

    for epoch in range(3):
        head.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}")
        for X, stage, ifng in pbar:
            X = X.to(DEVICE)
            stage = stage.to(DEVICE)
            ifng = ifng.to(DEVICE)
            with torch.no_grad():
                reps = enformer(X)  # (B, N, 1536)
                cls_emb = reps[:, 0, :]
            pred_stage, pred_ifng = head(cls_emb)
            loss = cls_loss(pred_stage, stage) + reg_loss(pred_ifng, ifng)
            optim.zero_grad(); loss.backward(); optim.step()
            pbar.set_postfix(loss=float(loss))

    torch.save(head.state_dict(), "head_pytorch.pt")
    print("✔ training done; head saved to head_pytorch.pt")


if __name__ == "__main__":
    main() 