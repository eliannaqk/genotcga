#!/usr/bin/env python
"""
Quick diagnostic script for Borzoi/Flashzoi embeddings on a *single* BAM file.

This re-uses the helper functions/constants defined in `train_borzoi.py` so the
exact same tiling / one-hot encoding / embedding logic is exercised, but avoids
all the DataLoader & training overhead.  Useful for debugging issues with tile
extraction or model inference before launching the full training run.

Example usage:
    python test_borzoi_one_bam.py \
           --bam /gpfs/.../TCGA-XYZ.bam.partial \
           --sample_id TCGA-XYZ \
           --model flashzoi \
           --cache /scratch/borzoi_cache.lmdb

The script will:
1. Extract all predefined tiles (see REGION & TILES in `train_borzoi.py`).
2. Compute embeddings for those tiles (optionally using the LMDB cache).
3. Print detailed progress messages at every step and optionally save the
   embeddings tensor to disk for inspection.
"""
import os, argparse, pickle, torch, pysam, time, lmdb
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Re-use everything from the original training script
# -----------------------------------------------------------------------------
from train_borzoi import (
    TILES, WIN, one_hot, fetch_tile, Embedder, Borzoi
)

# -----------------------------------------------------------------------------
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
            h = self.model.embed(t.unsqueeze(0).to('cuda'))  # (1,T,d)
            vec = h.mean(1).cpu()                            # (1,d)
            if self.env:
                with self.env.begin(write=True) as txn:
                    txn.put(key, pickle.dumps(vec.numpy(), protocol=4))
            outs.append(vec)
        return torch.cat(outs, dim=0)  

def timestamp() -> str:
    """Return current time in HH:MM:SS format – handy for long-running steps."""
    return time.strftime("%H:%M:%S")


def run(args):
    print(f"[{timestamp()}] Starting single-BAM diagnostic run")
    print(f"           BAM file : {args.bam}")
    print(f"           sample_id: {args.sample_id}")
    print(f"           backbone : {args.model}")
    print(f"           cache    : {args.cache or 'None'}")
    print("------------------------------------------------------------------")

    # ---------------------------------------------------------------------
    # Select and load backbone
    # ---------------------------------------------------------------------
    print(f"[{timestamp()}] Determining model checkpoint …")
    pid = (
        'johahi/flashzoi-replicate-0' if args.model == 'flashzoi'
        else 'johahi/borzoi-replicate-0'
    )
    print(f"[{timestamp()}] Loading pretrained weights: {pid}")
    borzoi = Borzoi.from_pretrained(pid).to('cuda')
    d_model = getattr(borzoi, 'dim', None) or getattr(borzoi, 'd_model', None) or getattr(borzoi, 'hidden_size', None) or getattr(getattr(borzoi, 'config', {}), 'hidden_size', 'unknown')
    print(f"[{timestamp()}] Model ready  (d_model = {d_model})")

    # ---------------------------------------------------------------------
    print(f"[{timestamp()}] Initialising embedder (cache = {bool(args.cache)}) …")
    # ensure parent directory of cache file exists
    if args.cache:
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    embedder = Embedder(borzoi, args.cache)
    print(f"[{timestamp()}] Embedder ready")

    # ---------------------------------------------------------------------
    # Tile extraction (always from reference FASTA)
    # ---------------------------------------------------------------------
    REF_PATH = "/home/eqk3/genotcga/data/hg38.fa"
    n_tiles = len(TILES)
    print(f"[{timestamp()}] Preparing to extract {n_tiles} tiles from FASTA …")

    ref = pysam.FastaFile(REF_PATH)

    tile_seqs = []
    for tile_ix, (chrom, start) in enumerate(tqdm(TILES, desc="tiles", unit="tile")):
        if tile_ix == 0:
            print(f"[{timestamp()}] First tile: {(chrom, start)} (reference)")
        seq = ref.fetch(chrom, start, start + WIN)
        tile_seqs.append(one_hot(seq))

    print(f"[{timestamp()}] Finished tile extraction from FASTA")

    # ---------------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------------
    print(f"[{timestamp()}] Computing embeddings …")
    embeddings = embedder(args.sample_id, tile_seqs)  # Shape: (tiles, d_model)
    print(f"[{timestamp()}] Embedding complete")
    print("------------------------------------------------------------------")
    print("✓ Embeddings computed:")
    print(f"   shape = {tuple(embeddings.shape)}  (tiles, d_model)")
    print(f"   dtype  = {embeddings.dtype}")

    # ---------------------------------------------------------------------
    # Optional saving
    # ---------------------------------------------------------------------
    if args.out:
        out_path = os.path.abspath(args.out)
        print(f"[{timestamp()}] Saving embeddings to {out_path} …")
        with open(out_path, "wb") as fh:
            pickle.dump(embeddings.numpy(), fh, protocol=4)
        print(f"[{timestamp()}] ✓ Saved embeddings")

    print("------------------------------------------------------------------")
    print(f"[{timestamp()}] Diagnostic run finished successfully")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bam', required=False, default="/gpfs/gibbs/project/gerstein/eqk3/tcgaData/3b7ba97d-6042-4f4a-8c5e-d37fba43e743/TCGA-63-A5MY-01A-11D-A92U-36.WholeGenome.RP-1657.bam.partial", help='Path to BAM(.partial) file')
    ap.add_argument('--sample_id', default='sample', help='Identifier used as key in the LMDB cache')
    ap.add_argument('--model', choices=['borzoi', 'flashzoi'], default='borzoi')
    ap.add_argument('--cache', default="/vast/palmer/scratch/gerstein/eqk3/my_borzoi_cache/borzoi_cache.lmdb", help='LMDB path to store/reuse embeddings')
    ap.add_argument('--out',   default=None, help='Optional pickle path to save the embeddings tensor')
    args = ap.parse_args()

    run(args)


if __name__ == '__main__':
    main()
