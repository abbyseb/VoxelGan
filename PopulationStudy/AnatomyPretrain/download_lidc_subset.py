#!/usr/bin/env python3
"""Download a CT-only LIDC-IDRI subset from TCIA (NBIA).

Full collection is ~128 GB / 1010 patients. Default is 40 patients, one
chest CT series each, so the anatomy-encoder pretrain can start without
filling the disk.

License: Creative Commons Attribution 3.0 Unported.
Collection DOI: https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
TCIA usage policy: https://wiki.cancerimagingarchive.net/x/c4hF
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

ROOT = Path(__file__).resolve().parent
META_DIR = ROOT / "data" / "lidc-idri" / "metadata"
DICOM_DIR = ROOT / "data" / "lidc-idri" / "dicom"
COLLECTION = "LIDC-IDRI"
SEED = 20260819


def _series_table(cache: Path) -> pd.DataFrame:
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        df = nbia.getSeries(collection=COLLECTION, modality="CT", format="df")
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    df["ImageCount"] = pd.to_numeric(df["ImageCount"], errors="coerce")
    df["FileSize"] = pd.to_numeric(df["FileSize"], errors="coerce")
    return df


def pick_subset(df: pd.DataFrame, n_patients: int, seed: int) -> pd.DataFrame:
    """One reasonably sized chest CT series per patient."""
    chest = df.copy()
    if "BodyPartExamined" in chest.columns:
        body = chest["BodyPartExamined"].fillna("").astype(str).str.upper()
        chest = chest[body.isin(["", "CHEST", "THORAX", "LUNG"])]
    chest = chest[(chest["ImageCount"] >= 80) & (chest["ImageCount"] <= 500)]
    chest = chest.sort_values(["PatientID", "ImageCount"], ascending=[True, False])
    one = chest.drop_duplicates("PatientID", keep="first")
    n = min(n_patients, len(one))
    subset = one.sample(n=n, random_state=seed).sort_values("PatientID")
    return subset.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patients", type=int, default=40)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--all", action="store_true", help="Download every CT series (~128 GB).")
    args = p.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    DICOM_DIR.mkdir(parents=True, exist_ok=True)
    cache = META_DIR / "lidc_idri_ct_series.csv"
    df = _series_table(cache)

    if args.all:
        subset = df
        tag = "all"
    else:
        subset = pick_subset(df, args.n_patients, args.seed)
        tag = f"n{len(subset)}_seed{args.seed}"

    manifest = META_DIR / f"lidc_subset_{tag}.csv"
    subset.to_csv(manifest, index=False)
    size_gb = float(subset["FileSize"].sum()) / 1e9
    print(
        f"Downloading {len(subset)} series from {subset['PatientID'].nunique()} patients "
        f"(~{size_gb:.1f} GB metadata size) -> {DICOM_DIR}"
    )
    print(f"Manifest: {manifest}")

    nbia.downloadSeries(
        subset["SeriesInstanceUID"].tolist(),
        path=str(DICOM_DIR),
        input_type="list",
        max_workers=args.max_workers,
        csv_filename=str(META_DIR / f"lidc_download_{tag}"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
