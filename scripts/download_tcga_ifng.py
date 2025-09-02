#!/usr/bin/env python3
# download_tcga_ifng.py
"""
Download TCGA-SKCM (melanoma) WGS BAMs plus corresponding RNA-seq
HTSeq-FPKM-UQ files needed to build IFN-γ gene-expression scores.

Requires:
  • requests   (pip/conda install requests)
  • gdc-client in $PATH (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

import requests
import pprint, textwrap
pp = pprint.PrettyPrinter(indent=2, width=120).pprint

GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-SKCM"

IFNG_GENES = [
    "CD3D","IDO1","CIITA","CD3E","CCL5","GZMK","CD2","HLA-DRA","CXCL13","IL2RG",
    "NKG7","HLA-E","CXCR6","LAG3","TAGAP","CXCL10","STAT1","GZMB"
]

##############################################################################
# Helper functions
##############################################################################


def api_post(endpoint: str, payload: Dict, page_size: int = 5000) -> List[Dict]:
    """POST to a GDC endpoint, automatically handling paging."""
    url = f"{GDC_API}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    results = []
    while True:
        payload["size"] = page_size
        resp = requests.post(url, headers=headers, data=json.dumps(payload))
        resp.raise_for_status()
        data = resp.json()["data"]
        results.extend(data["hits"])
        if len(results) >= data["pagination"]["total"]:
            break
        payload["from"] = payload.get("from", 0) + page_size
    return results


def build_filter(**kwargs):
    """Utility to build a simple 'in' filter."""
    return {"op": "in", "content": {"field": kwargs["field"], "value": kwargs["value"]}}


def get_case_file_map(max_samples) -> Dict[str, Dict[str, str]]:
    """
    Returns {case_id: {'wgs': file_id, 'rna': file_id}}
    only for cases where BOTH file types exist.
    """
    # ----- 1. query WGS BAMs -------------------------------------------------
    wgs_filter = {
        "op": "and",
        "content": [
            build_filter(field="cases.project.program.name", value=["TCGA"]),
            build_filter(field="files.experimental_strategy", value=["WGS"]),
            build_filter(field="files.data_type", value=["Aligned Reads"])
        ],
    }
    wgs_files = api_post(
        "files",
        {
            "filters": wgs_filter,
            "fields": "file_id,cases.case_id",
            "format": "JSON",
        },
    )
    print("\n[DEBUG] total WGS hits:", len(wgs_files))
    if wgs_files:
        print("  first 5:", [hit["cases"][0]["case_id"] for hit in wgs_files[:5]])
    case2wgs = {hit["cases"][0]["case_id"]: hit["file_id"] for hit in wgs_files}

    # ----- 2. query RNA-seq HTSeq FPKM-UQ ------------------------------------
    rna_filter = {
        "op": "and",
        "content": [
            build_filter(field="cases.project.program.name", value=["TCGA"]),
            build_filter(field="files.data_category", value=["Transcriptome Profiling"]),
            build_filter(field="files.data_type", value=["Gene Expression Quantification"]),
            build_filter(field="files.analysis.workflow_type", value=[
                "HTSeq - FPKM-UQ",
                "HTSeq - Counts",
                "STAR - Counts"
            ]),
        ],
    }
    rna_files = api_post(
        "files",
        {
            "filters": rna_filter,
            "fields": "file_id,cases.case_id",
            "format": "JSON",
        },
    )
    print("\n[DEBUG] total RNA hits:", len(rna_files))
    if rna_files:
        print("  first 5:", [hit["cases"][0]["case_id"] for hit in rna_files[:5]])
    case2rna = {hit["cases"][0]["case_id"]: hit["file_id"] for hit in rna_files}

    # ----- 3. intersect ------------------------------------------------------
    common_cases = [c for c in case2wgs if c in case2rna]
    print("\n[DEBUG] common cases:", len(common_cases))
    if max_samples != "all":
        common_cases = common_cases[: int(max_samples)]

    return {
        c: {"wgs": case2wgs[c], "rna": case2rna[c]}
        for c in common_cases
    }


def write_manifest(case_map: Dict[str, Dict[str, str]], out_path: Path):
    """Write GDC manifest TSV."""
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["id", "filename"])  # gdc-client only cares about 'id'
        for cdict in case_map.values():
            for fid in cdict.values():
                writer.writerow([fid, ""])
    print(f"Manifest written to {out_path}  ({sum(len(v) for v in case_map.values())} files)")


def run_gdc_client(manifest_path: Path, token_path: Path, out_dir: Path):
    cmd = [
        "gdc-client",
        "download",
        "--token", str(token_path),
        "--manifest", str(manifest_path),
        "--dir", str(out_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


##############################################################################
# Main
##############################################################################
def main():
    p = argparse.ArgumentParser(description="Download TCGA-SKCM WGS + RNA files")
    p.add_argument("--token", required=True, type=Path, help="Path to GDC user token file")
    p.add_argument("--max-samples", default="10",
                   help="'all' or N (default 10) samples to download")
    p.add_argument("--outdir", default=Path("/gpfs/gibbs/project/gerstein/eqk3/tcgaData"), type=Path,
                   help="Directory to store downloaded files (default: /gpfs/.../tcgaData)")
    args = p.parse_args()

    case_map = get_case_file_map(args.max_samples)
    if not case_map:
        sys.exit("No cases with both WGS and RNA files found!")

    args.outdir.mkdir(exist_ok=True)
    manifest_path = args.outdir / "gdc_manifest.tsv"
    write_manifest(case_map, manifest_path)

    run_gdc_client(manifest_path, args.token, args.outdir)

    print("\nNext steps:")
    print("1. For RNA files (HTSeq-FPKM-UQ), load each tsv and extract the"
          f" {len(IFNG_GENES)} IFN-γ genes:")
    print("     genes_of_interest =", IFNG_GENES)
    print("2. Centre-scale within cohort, then compute your chosen IFN-γ score metric.")


if __name__ == "__main__":
    main()