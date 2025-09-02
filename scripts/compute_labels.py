#!/usr/bin/env python3
"""
compute_labels.py
=================
Builds the `labels.csv` required by the PyTorch training scripts.

For every RNA-seq quantification file downloaded by `download_tcga_ifng.py` it
performs three tasks:

1. Extract the *sample / case id* and corresponding GDC **file UUID**.
2. Call the GDC REST API to obtain the tumour stage for that case.
3. Parse the expression TSV to compute an IFN-γ signature score – the arithmetic
   mean of log2(expression + 1) across the 18 genes used in Long *et al.* 2022
   [Genome Medicine](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-022-01024-y "IFN-γ signature").

Output
------
`labels.csv` with columns:
    sample_id,tumor_stage,ifng

Assumptions
-----------
• RNA TSVs live under `gdc_downloads/` (same as created by the download script).
• Filenames are `<uuid>.tsv` (or `.tsv.gz`). The UUID is used to query the GDC.
• The TSV has either 2 columns (gene_id, value) **or** ≥3 columns where the 2nd
  column is `gene_name`.
• Expressions are FPKM-UQ or counts – we just take the numeric column.

Install the extra dependency once:
    pip install mygene tqdm requests pandas numpy
"""

# flake8: noqa
from __future__ import annotations

import gzip, json, re, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Mandatory dependency – abort if unavailable
# ---------------------------------------------------------------------------
try:
    import mygene  # type: ignore

    _mg = mygene.MyGeneInfo()
except ImportError as e:
    sys.exit("❌ The 'mygene' package is required. Install with `pip install mygene`. ")


# ---------------------------------------------------------------------------
# Ensembl → HGNC symbol cache
# ---------------------------------------------------------------------------
_ens_cache: Dict[str, str] = {}


def ensembl_to_symbol(ens_id: str) -> str | None:
    """Return official gene symbol for an Ensembl gene ID (strip version suffix)."""
    ens_id_clean = ens_id.split(".")[0]
    if ens_id_clean in _ens_cache:
        return _ens_cache[ens_id_clean]
    if _mg is None:
        return None
    try:
        res = _mg.query(f"ensembl.gene:{ens_id_clean}", fields="symbol", size=1)
        hits = res.get("hits", [])
        if hits:
            sym = hits[0].get("symbol")
            if sym:
                _ens_cache[ens_id_clean] = sym.upper()
                return _ens_cache[ens_id_clean]
    except Exception:
        pass
    return None


###############################################################################
# Constants
###############################################################################

GDC_API = "https://api.gdc.cancer.gov"
RNA_DIR = Path("gdc_downloads")
IFNG_GENES = [
    "CD3D","IDO1","CIITA","CD3E","CCL5","GZMK","CD2","HLA-DRA","CXCL13","IL2RG",
    "NKG7","HLA-E","CXCR6","LAG3","TAGAP","CXCL10","STAT1","GZMB"
]

###############################################################################
# Helpers
###############################################################################

def open_tsv(path: Path):
    """Return iterator over lines, transparently handling .gz files."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_expression(path: Path) -> Dict[str, float]:
    """Return mapping gene_symbol → expression value for a single sample."""
    expr: Dict[str, float] = {}
    header_cols: List[str] | None = None
    target_idx: int | None = None  # index of the expression column we want

    with open_tsv(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            parts = line.rstrip().split()  # split on any whitespace (tab or space)
            if header_cols is None and parts and parts[0] == "gene_id":
                # This is the header line produced by STAR quantification
                header_cols = parts
                # Prefer FPKM-UQ if present, else TPM, else 4th column (raw counts)
                for col_name in ["fpkm_uq_unstranded", "tpm_unstranded"]:
                    if col_name in header_cols:
                        target_idx = header_cols.index(col_name)
                        break
                if target_idx is None:
                    target_idx = len(header_cols) - 1  # fall back to last column
                continue  # skip header line

            if len(parts) < 2:
                continue

            if len(parts) == 2:  # HTSeq two-column format
                gene_id, val = parts
                gene_symbol = ensembl_to_symbol(gene_id) or ""
            else:
                gene_id = parts[0]
                gene_symbol = parts[1]
                val = parts[target_idx] if target_idx is not None and target_idx < len(parts) else parts[2]

            if not gene_symbol:
                continue
            try:
                expr[gene_symbol.upper()] = float(val)
            except ValueError:
                continue

    return expr


def compute_ifng_score(expr: Dict[str, float]) -> float:
    values = [expr.get(g, 0.0) for g in IFNG_GENES]
    # log2(x+1)
    log_vals = [np.log2(v + 1.0) for v in values]
    return float(np.mean(log_vals))


def fetch_stage_api(file_uuid: str) -> Tuple[str, str]:
    """Return (case_id, tumor_stage) via GDC REST API. Can raise HTTPError."""
    fields = "cases.case_id,cases.diagnoses.tumor_stage"
    url = f"{GDC_API}/files/{file_uuid}?fields={fields}"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()["data"]
    case_id = data["cases"][0]["case_id"]
    diagnoses = data["cases"][0].get("diagnoses", [])
    stage = "unknown"
    for diag in diagnoses:
        stage_val = diag.get("tumor_stage")
        if stage_val:
            stage = stage_val
            break
    return case_id, stage


def fetch_stage_local(meta_path: Path) -> Tuple[str, str]:
    """Extract (case_id, tumor_stage) from a metadata.json file downloaded by gdc-client."""
    try:
        with meta_path.open() as fh:
            data = json.load(fh)
    except Exception:
        raise FileNotFoundError("metadata.json missing or unreadable")

    cases = data.get("cases", [])
    if not cases:
        raise ValueError("No case information in metadata.json")
    case_id = cases[0].get("case_id", "unknown_case")
    diagnoses = cases[0].get("diagnoses", [])
    stage = "unknown"
    for diag in diagnoses:
        stage_val = diag.get("tumor_stage")
        if stage_val:
            stage = stage_val
            break
    return case_id, stage

###############################################################################
# Main
###############################################################################

def main():
    """Entry point – builds labels.csv from RNA-seq quantifications."""

    # ------------------------------------------------------------------ #
    # Fixed paths – adjust here if your data lives elsewhere
    # ------------------------------------------------------------------ #
    rna_dir = Path("/gpfs/gibbs/project/gerstein/eqk3/tcgaData")
    manifest_path = rna_dir / "gdc_manifest.tsv"

    if not manifest_path.exists():
        sys.exit(f"Manifest file not found: {manifest_path}")

    if not rna_dir.exists():
        sys.exit(f"RNA directory {rna_dir} not found – check the path.")

    print(f"📂 RNA directory: {rna_dir}")
    print(f"📄 Manifest:      {manifest_path}")

    # ------------------------------------------------------------------ #
    # Collect RNA-seq TSVs using manifest only (one row per UUID)
    # ------------------------------------------------------------------ #
    tsv_entries: List[Tuple[str, Path]] = []

    with manifest_path.open() as fh:
        next(fh)  # skip header line
        for line in fh:
            fid = line.strip().split("\t")[0]
            if not fid:
                continue
            sample_dir = rna_dir / fid
            tsv_files = list(sample_dir.glob("*.tsv*"))
            if not tsv_files:
                print(f"⚠ No TSV in {sample_dir}; skipping.")
                continue
            tsv_entries.append((fid, tsv_files[0]))


    print(f"🗂 Found {len(tsv_entries)} RNA-seq TSV files to process\n")

    if not tsv_entries:
        sys.exit("No RNA-seq TSV files found – aborting.")

    rows = []
    for fid, path in tqdm(tsv_entries, desc="processing RNA files"):
        metadata_json = (path.parent / "metadata.json")
        try:
            if metadata_json.exists():
                case_id, stage = fetch_stage_local(metadata_json)
            else:
                # fall back to REST API
                case_id, stage = fetch_stage_api(fid)
        except Exception as e:
            print(f"✗ metadata fetch failed for {fid}: {e}")
            continue
        expr = parse_expression(path)
        score = compute_ifng_score(expr)
        missing = [g for g in IFNG_GENES if g not in expr]
        print(f"✔ {fid}  case_id={case_id}  stage={stage}  IFNγ={score:.3f}  missing={len(missing)} genes")
        if missing:
            print("   ↪ Missing genes:", ", ".join(missing))

        rows.append({"sample_id": case_id, "tumor_stage": stage, "ifng": score})

    if not rows:
        sys.exit("No valid RNA files processed – aborting.")

    df = pd.DataFrame(rows).drop_duplicates("sample_id")
    df.to_csv("labels.csv", index=False)
    print(f"✔ Wrote labels.csv with {len(df)} samples")


if __name__ == "__main__":
    main() 