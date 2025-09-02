#!/usr/bin/env python3
"""
precompute_embeddings_and_train_head.py
--------------------------------------
1. Iterates over BAM files in ~/tcgaData/, sliding 196 608-bp windows with
   50-kb stride, passes them through a *frozen* Enformer trunk and averages the
   CLS-token embedding across all windows to yield one 1536-d vector per sample.
2. Caches embeddings under emb/<sample>.pt (created if not present).
3. Loads all cached embeddings, builds a simple MLP head and trains it for the
   dual tasks tumour-stage (classification) and IFN-γ score (regression).

Run:
    python scripts/precompute_embeddings_and_train_head.py

Dependencies: torch, enformer-pytorch, pysam, pandas, tqdm
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pysam
from tqdm import tqdm

try:
    from enformer_pytorch import Enformer
except ImportError:
    sys.exit("❌ enformer-pytorch not installed. `pip install enformer-pytorch`. ")

WINDOW = 196_608
STRIDE = 50_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUC = {"A":0, "C":1, "G":2, "T":3}

def one_hot(seq: str):
    a = torch.zeros((len(seq),4), dtype=torch.float32)
    for i,b in enumerate(seq.upper()):
        idx = NUC.get(b)
        if idx is not None:
            a[i,idx]=1.
    return a

def windows(length):
    i=0
    while i+WINDOW<=length:
        yield i,i+WINDOW
        i+=STRIDE

###############################################################################
# Step 1 – compute & cache embeddings
###############################################################################

def compute_embeddings():
    root = Path.home()/"tcgaData"
    bam_files = sorted(root.glob("*.bam"))
    if not bam_files:
        sys.exit(f"No BAM files in {root}")
    emb_dir = Path("emb"); emb_dir.mkdir(exist_ok=True)
    enformer = Enformer.from_pretrained("EleutherAI/enformer-pytorch").eval().to(DEVICE)
    for p in enformer.parameters():
        p.requires_grad=False
    ref = pysam.FastaFile(str(Path.home()/"reference/GRCh38.fa"))
    for bam in bam_files:
        out_path = emb_dir/f"{bam.stem}.pt"
        if out_path.exists():
            continue  # cached
        with pysam.AlignmentFile(bam,"rb") as b:
            acc=None; n=0
            for chr_ in b.references:
                L=b.get_reference_length(chr_)
                for s,e in windows(L):
                    seq=ref.fetch(chr_,s,e)
                    x=one_hot(seq).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        reps=enformer(x)
                        cls=reps[:,0,:].squeeze(0)
                    acc = cls if acc is None else acc+cls
                    n+=1
            mean_emb = acc/ n
            torch.save(mean_emb.cpu(), out_path)
            print(f"saved {out_path}  (n={n} windows)")

###############################################################################
# Step 2 – dataset & training on frozen embeddings
###############################################################################

class EmbDS(Dataset):
    def __init__(self, emb_dir: Path, labels: pd.DataFrame):
        self.paths = list(emb_dir.glob("*.pt"))
        self.labels = labels.set_index("sample_id")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]
        emb=torch.load(p)
        row=self.labels.loc[p.stem]
        return emb, torch.tensor(row.tumor_stage, dtype=torch.long), torch.tensor(row.ifng, dtype=torch.float32)

def train_head():
    emb_dir=Path("emb")
    labels=pd.read_csv("labels.csv")
    ds=EmbDS(emb_dir,labels)
    dl=DataLoader(ds,batch_size=8,shuffle=True)
    head=nn.Sequential(nn.Linear(1536,256), nn.ReLU(), nn.Dropout(0.2),
                       nn.Linear(256,4+1))  # first 4 = stage, last 1 = ifng
    head.to(DEVICE)
    opt=torch.optim.Adam(head.parameters(), lr=1e-3)
    cls_loss=nn.CrossEntropyLoss(); reg_loss=nn.MSELoss()
    for epoch in range(20):
        head.train(); tot=0; n=0
        for emb,stage,ifng in dl:
            emb,stage,ifng=emb.to(DEVICE),stage.to(DEVICE),ifng.to(DEVICE)
            out=head(emb)
            pred_stage=out[:,:4]
            pred_ifng=out[:,4]
            loss=cls_loss(pred_stage,stage)+reg_loss(pred_ifng,ifng)
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item(); n+=1
        print(f"epoch {epoch+1}: loss {tot/n:.4f}")
    torch.save(head.state_dict(),"head_from_embeddings.pt")
    print("✔ head saved to head_from_embeddings.pt")

###############################################################################

if __name__=="__main__":
    compute_embeddings()
    train_head() 