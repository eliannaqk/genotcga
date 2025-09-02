#!/usr/bin/env python
"""
train_ifng_multi.py – fine‑tune a small head on Borzoi/Flashzoi embeddings
to predict continuous IFN‑γ from TCGA BAMs.

Example:
  python train_ifng_multi.py \
         --bam_dir /gpfs/.../tcgaData \
         --labels labels.csv \
         --model flashzoi \
         --aggregation attn \
         --cache /vast/palmer/scratch/gerstein/eqk3/my_borzoi_cache/borzoi_cache.lmdb
"""

import os, argparse, random, hashlib, pickle, lmdb, pysam
import torch, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from borzoi_pytorch import Borzoi                    # same class for both variants

# ---------- Reference FASTA ---------------------------------------------------
REF_PATH = "/home/eqk3/genotcga/data/hg38.fa"

# ---------- Genomic tiling ---------------------------------------------------

# REGION = ("chr2", 163_455_290, 166_962_322)          # 2q24.3 (hg38): IFIH1 + SCN cluster
# WIN    = 196_608
# STRIDE = 64_000

# def make_tiles(chrom, start, end, win=WIN, stride=STRIDE):
#     stops = end - win
#     return [(chrom, s) for s in range(start, stops + 1, stride)]
# TILES = make_tiles(*REGION)
WIN = 196_608                                  # Borzoi receptive field
IFIH1_CENTER = 162_292_879                     # hg38 midpoint of IFIH1
VIEW_START   = IFIH1_CENTER - WIN // 2         # 162 194 575
VIEW_TILE    = ("chr2", VIEW_START)            # single-element tuple list

TILES = [VIEW_TILE] 
# ---------- DNA helpers ------------------------------------------------------

def one_hot(seq: str) -> torch.Tensor:
    mapping = {'A':0,'C':1,'G':2,'T':3}
    arr = torch.zeros(4, len(seq), dtype=torch.float16)   # fp16 to save RAM
    for i,b in enumerate(seq.upper()):
        idx = mapping.get(b)
        if idx is not None: arr[idx,i] = 1.
    return arr

def fetch_tile(bam, chrom, start):
    return bam.fetch(chrom, start, start + WIN).get_reference_sequence()

# ---------- Dataset / Dataloader --------------------------------------------

class BamTileSet(Dataset):
    """Returns (sample_id, list[one‑hot tensors]) per TCGA BAM."""
    def __init__(self, csv, ref_path=REF_PATH):
        meta = pd.read_csv(csv)
        self.ids = meta['sample_id'].tolist()
        self.ref = pysam.FastaFile(ref_path)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        tiles = []
        for chrom, start in TILES:
            seq = self.ref.fetch(chrom, start, start + WIN)
            tiles.append(one_hot(seq))
        return sid, tiles

def collate(batch):
    # keep list structure; each sample may have many tiles
    sids, tile_lists = zip(*batch)
    return sids, tile_lists

# ---------- Embedding with LMDB cache ---------------------------------------

class Embedder:
    def __init__(self, model, cache_path=None):
        self.model = model.eval().requires_grad_(False)

        if cache_path:
            # Ensure parent directory of the LMDB file exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # Open (or create) the LMDB environment backing file
            self.env = lmdb.open(cache_path, map_size=2_000_000_000_000, subdir=False)
        else:
            self.env = None

    def _key(self, sid, idx):
        return f"{sid}:{idx}".encode()

    def __call__(self, sid, tiles):            # tiles = list[(4,L)]
        outs = []
        for i,t in enumerate(tiles):
            key = self._key(sid, i)
            if self.env:
                with self.env.begin() as txn:
                    buf = txn.get(key)
                if buf:
                    outs.append(torch.tensor(pickle.loads(buf)))
                    continue
            preds,h = self.model.forward(t.unsqueeze(0).to('cuda'), return_embeddings = True)  # (1,T,d)
            #h = self.model.forward_features(t.unsqueeze(0).to('cuda'))  # (1,T,d)
            vec = h.mean(1).cpu()                            # (1,d)
            if self.env:
                with self.env.begin(write=True) as txn:
                    txn.put(key, pickle.dumps(vec.numpy(), protocol=4))
            outs.append(vec)
        return torch.cat(outs, dim=0)          # (tiles, d)

# ---------- Aggregation heads -----------------------------------------------

class ConcatMLP(nn.Module):
    def __init__(self, d_tile, n_tiles):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_tile * n_tiles, 512),
                                 nn.GELU(),
                                 nn.Linear(512, 1))
    def forward(self, reps):                   # reps: (B, tiles, d)
        flat = reps.flatten(1)                 # (B, tiles*d)
        return self.mlp(flat).squeeze(-1)

class TileAttention(nn.Module):
    def __init__(self, d, tiles, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp  = nn.Sequential(nn.Linear(d,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, reps):                   # (B, tiles, d)
        h,_ = self.attn(reps,reps,reps)
        h   = self.norm(h).mean(1)             # attentively pooled (B,d)
        return self.mlp(h).squeeze(-1)

def build_head(agg, d, tiles):
    return ConcatMLP(d, tiles) if agg=='concat' else TileAttention(d, tiles)

# ---------- Training loop ----------------------------------------------------

def train(args):
    # backbone choice
    pid = ('johahi/flashzoi-replicate-0' if args.model=='flashzoi'
           else 'johahi/borzoi-replicate-0')
    borzoi = Borzoi.from_pretrained(pid).to('cuda')
    print(f"borzoi: {borzoi}")
    print(f"borzoi.encoder: {borzoi.encoder}")
    with torch.no_grad(), torch.cuda.amp.autocast():
        dummy = torch.zeros(1, 4, WIN, device='cuda', dtype=torch.float16)
        preds,h = borzoi.forward(dummy, return_embeddings = True) 
        d_model = h.shape[-1]
        print(f"d_model: {d_model}")

    ds  = BamTileSet(args.labels)
    dl  = DataLoader(ds, batch_size=args.batch, shuffle=True,
                     num_workers=4, collate_fn=collate)

    embed = Embedder(borzoi, args.cache)
    head  = build_head(args.aggregation, d_model, len(TILES)).to('cuda')

    opt   = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler= torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        head.train()
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for sids, tile_lists in pbar:
            reps = [embed(sid, tiles) for sid,tiles in zip(sids, tile_lists)]
            reps = torch.stack(reps).to('cuda')        # (B, tiles, d)
            labels= torch.tensor([float(0)]*len(sids)).to('cuda')  # dummy; plug real labels if needed

            with torch.cuda.amp.autocast():
                pred = head(reps)
                loss = nn.functional.mse_loss(pred, labels)

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=loss.item())

    torch.save({'head': head.state_dict(),
                'tiles': len(TILES),
                'd': d_model,
                'agg': args.aggregation},
               args.out)
    print(f"✓ saved {args.out}")

# ---------- CLI --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bam_dir', default=None, help='unused (reference FASTA is used instead)')
    ap.add_argument('--labels',  required=True)
    ap.add_argument('--model', choices=['borzoi','flashzoi'], default='borzoi')
    ap.add_argument('--aggregation', choices=['concat','attn'], default='concat')
    ap.add_argument('--cache', default="/vast/palmer/scratch/gerstein/eqk3/my_borzoi_cache/borzoi_cache.lmdb",
                    help="LMDB path to store/reuse embeddings")
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch',  type=int, default=2)
    ap.add_argument('--out',    default='ifng_head_multi.pt')
    args = ap.parse_args()
    train(args)

if __name__ == "__main__":
    main()