#!/usr/bin/env python3
"""Build an ADNI intermediate task that matches the RCT target task structurally.

WHY THIS EXISTS
---------------
The RCT target is: two scans per subject (T1, T2) -> change in a cognitive composite
between them. The existing ADNI stage-1 task is: one scan -> diagnosis at that visit.
Those differ in input structure (1 vs 2 scans), label semantics (state vs change) and
construct (diagnosis vs memory/executive composite). Any ADNI->RCT transfer result from
the single-scan task is therefore confounded with task mismatch, and "the intermediate
task was wrong" remains an unfalsified explanation for a weak transfer.

ADNI can supply the matched task and it has never been used here:

    ADNI_MEM  <-> the RCT memory composite      (delay recall)
    ADNI_EF   <-> the RCT executive composite

Output format is deliberately identical to `data/rct/<cohort>/bold_T1T2.pt`, i.e.
[N, 424, T, 2] with NaN padding, so `datasets.T12DualDataset` and every downstream
entrypoint work unchanged on it.

PAIR SELECTION
--------------
`--pairs first_last` (default) takes one pair per subject: the earliest and latest scanned
visit that both carry the outcome. One row per subject, so any split is leakage-free.
`--pairs all` takes every ordered pair, which multiplies the data but puts the same subject
in many rows -- callers must then group by PTID when splitting. The count of each is
printed so the choice is visible in the log.

CONFOUND CONTROL
----------------
`--same_protocol` (default on) requires both scans of a pair to come from the same
acquisition protocol group. This is not optional hygiene: the per-ROI amplitude profile
predicts protocol group at AUC 0.97 (the original cohort audit), so a pair that crosses
protocols has a change signal contaminated by a scanner switch that dwarfs two years of
cognitive change.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from paths import ADNI_COG_CSV, ADNI_COG_NEW_CSV, ADNI_SCAN_DIR_RAW, DERIVED_DIR  # noqa: E402
from prepare_adni import native_tr, protocol_group, resample_to_tr, session_stem  # noqa: E402

#: Outcome columns and the construct each one matches in the RCTs.
OUTCOMES = {
    "mem": ("ADNI_MEM", "RCT memory composite (delay recall)"),
    "ef": ("ADNI_EF", "RCT executive composite"),
}

SES = re.compile(r"_ses-M(\d+)")


def visit_month(stem: str) -> int:
    m = SES.search(stem)
    if not m:
        raise ValueError(f"cannot read a visit month from {stem!r}")
    return int(m.group(1))


def build_table(scan_dir: Path) -> pd.DataFrame:
    """One row per scan that has a cognitive row: PTID, month, path, outcomes."""
    with_files = pd.read_csv(ADNI_COG_CSV, low_memory=False)
    with_dx = pd.read_csv(ADNI_COG_NEW_CSV, low_memory=False)
    if len(with_files) != len(with_dx):
        raise SystemExit("the two cognitive CSVs no longer align row-for-row")
    aligned = with_files["ID"].str.replace("sub-ADNI", "", regex=False).values == with_dx[
        "PTID"].str.replace("_", "", regex=False).values
    if not aligned.all():
        raise SystemExit(f"row-wise ID mismatch in {(~aligned).sum()} rows; refusing to guess")
    cog = with_dx.assign(FILE_NAME=with_files["FILE_NAME"].values)

    scans = sorted(scan_dir.glob("*.pt"))
    if not scans:
        raise SystemExit(f"no .pt scans under {scan_dir}")
    avail = {session_stem(p): p for p in scans}
    if len(avail) != len(scans):
        raise SystemExit(f"{scan_dir}: session stems collide after stripping the cohort suffix")

    df = cog[cog["FILE_NAME"].isin(avail)].drop_duplicates("FILE_NAME").copy()
    df["path"] = df["FILE_NAME"].map(avail)
    df["month"] = df["FILE_NAME"].map(visit_month)
    print(f"{len(scans)} scans in {scan_dir.name}; {len(df)} have a cognitive row "
          f"over {df['PTID'].nunique()} subjects")
    return df


def load_scan(path: Path, tlen: int, target_tr: float | None, max_dead_roi: int):
    """Return (signal[424, tlen], raw_len, gain) or (None, reason, None)."""
    sig = torch.load(path, map_location="cpu", weights_only=False).float()
    if sig.shape[0] != C.NUM_BRAIN_VOXELS:
        raise SystemExit(f"{path.name}: expected {C.NUM_BRAIN_VOXELS} ROIs, got {sig.shape[0]}")
    raw_t = int(sig.shape[1])
    min_raw = tlen if not target_tr else int(np.ceil(tlen * target_tr / native_tr(raw_t)))
    if raw_t < min_raw:
        return None, "too_short", None
    if int((sig.std(dim=1) < 1e-6).sum()) > max_dead_roi:
        return None, "dead_roi", None
    if target_tr:
        sig = resample_to_tr(sig, native_tr(raw_t), target_tr)
        if sig.shape[1] < tlen:
            return None, "too_short_after_resample", None
    sig = sig[:, :tlen]
    sig = sig - sig.mean(1, keepdim=True)
    gain = float(sig.std(1, unbiased=False).median())
    return sig, raw_t, gain


def main():
    ap = argparse.ArgumentParser(description="Build the ADNI paired (T1,T2) -> change task")
    ap.add_argument("--scan_dir", default=str(ADNI_SCAN_DIR_RAW))
    ap.add_argument("--pairs", default="first_last", choices=("first_last", "all"))
    ap.add_argument("--tlen", type=int, default=140, help="crop length in frames; multiple of 20")
    ap.add_argument("--target_tr", type=float, default=None)
    ap.add_argument("--min_interval", type=int, default=6,
                    help="minimum months between the two scans of a pair")
    ap.add_argument("--same_protocol", dest="same_protocol", action="store_true", default=True)
    ap.add_argument("--allow_cross_protocol", dest="same_protocol", action="store_false",
                    help="permit pairs whose two scans come from different protocol groups. "
                         "The amplitude profile predicts protocol at AUC 0.97, so the change "
                         "signal then carries a scanner switch; off by default for that reason.")
    ap.add_argument("--amplitude", default="official",
                    choices=("zscore", "preserve", "official"),
                    help="official is BrainLM's own recipe -- per-ROI mean-centring and no "
                         "division at all. Measured on CN/AD it beats the per-scan-scalar "
                         "`preserve` variant on every readout (emb_fc 0.6846 vs 0.6607), "
                         "because `preserve` removes between-subject amplitude that the "
                         "encoder uses. See prepare_adni.py for the full comparison.")
    ap.add_argument("--max_scale_dev", type=float, default=3.0)
    ap.add_argument("--max_dead_roi", type=int, default=0)
    ap.add_argument("--out_dir", default=str(DERIVED_DIR / "adni_paired"))
    args = ap.parse_args()

    if args.tlen % C.TIMEPOINT_PATCHING_SIZE:
        raise SystemExit(f"--tlen must be a multiple of {C.TIMEPOINT_PATCHING_SIZE}")

    scan_dir = Path(args.scan_dir)
    df = build_table(scan_dir)
    have = [k for k, (col, _) in OUTCOMES.items() if col in df.columns]
    if not have:
        raise SystemExit(f"none of {[c for c, _ in OUTCOMES.values()]} is in the cognitive table")
    print(f"outcomes available: {[(k, OUTCOMES[k][0]) for k in have]}")

    # Enumerate candidate pairs before touching any imaging, so the reasons a pair is
    # rejected are separable from the reasons a scan is rejected.
    cand, why = [], {"one_scan": 0, "interval": 0, "cross_protocol": 0, "no_outcome": 0}
    for ptid, g in df.groupby("PTID"):
        g = g.sort_values("month")
        if len(g) < 2:
            why["one_scan"] += 1
            continue
        idx = list(range(len(g)))
        pool = [(idx[0], idx[-1])] if args.pairs == "first_last" else [
            (i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        for i, j in pool:
            r1, r2 = g.iloc[i], g.iloc[j]
            if r2["month"] - r1["month"] < args.min_interval:
                why["interval"] += 1
                continue
            if not any(pd.notna(r1[OUTCOMES[k][0]]) and pd.notna(r2[OUTCOMES[k][0]])
                       for k in have):
                why["no_outcome"] += 1
                continue
            cand.append((ptid, r1, r2))
    print(f"candidate pairs: {len(cand)}   rejected: {why}")

    keep, drop = [], {"too_short": 0, "dead_roi": 0, "too_short_after_resample": 0,
                      "cross_protocol": 0}
    cache: dict = {}
    for ptid, r1, r2 in cand:
        out = []
        for r in (r1, r2):
            if r["FILE_NAME"] not in cache:
                cache[r["FILE_NAME"]] = load_scan(Path(r["path"]), args.tlen,
                                                  args.target_tr, args.max_dead_roi)
            out.append(cache[r["FILE_NAME"]])
        if any(o[0] is None for o in out):
            drop[next(o[1] for o in out if o[0] is None)] += 1
            continue
        p1, p2 = protocol_group(int(o1_raw := out[0][1])), protocol_group(int(out[1][1]))
        if args.same_protocol and p1 != p2:
            drop["cross_protocol"] += 1
            continue
        keep.append((ptid, r1, r2, out[0], out[1], p1, p2))
    print(f"pairs surviving imaging QC: {len(keep)}   dropped: {drop}")
    if not keep:
        raise SystemExit("no pairs survived; loosen --min_interval or --allow_cross_protocol")

    # Off-scale gate, on the pair: if either scan is off-scale the pair goes. Same
    # criterion as prepare_adni.py, applied in both amplitude modes so the two builds
    # differ only in the amplitude treatment.
    gains = np.array([[k[3][2], k[4][2]] for k in keep])
    ref = float(np.median(gains))
    bad = (np.abs(np.log2(np.maximum(gains, 1e-12) / ref)) > args.max_scale_dev).any(axis=1)
    if bad.any():
        print(f"dropped {int(bad.sum())} pairs with an off-scale scan "
              f"(cohort median per-ROI sd {ref:.4f}, gate +/-{args.max_scale_dev:g} log2)")
    keep = [k for k, b in zip(keep, bad) if not b]

    if args.amplitude in ("preserve", "official"):
        spread = max(float(k[3][0].std(1, unbiased=False).max()
                           / k[3][0].std(1, unbiased=False).median().clamp_min(1e-12))
                     for k in keep[:20])
        if spread < 1.01:
            raise SystemExit(
                f"--amplitude preserve on release '{scan_dir.name}': per-ROI sds are within "
                f"{(spread - 1) * 100:.3f}% of the median, so this release is already "
                "z-scored and has no amplitude to preserve.")
        print(f"amplitude check: max/median per-ROI sd reaches {spread:.3f}")

    def norm(sig, gain):
        if args.amplitude == "official":
            return sig                                          # already mean-centred; upstream divides by nothing
        if args.amplitude == "preserve":
            return sig / max(gain, 1e-12)                      # one scalar for the scan
        return sig / sig.std(1, keepdim=True).clamp_min(1e-6)   # 424 scalars: flattens ratios

    x = torch.stack([torch.stack([norm(k[3][0], k[3][2]), norm(k[4][0], k[4][2])], dim=-1)
                     for k in keep])                            # [N, 424, tlen, 2]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{scan_dir.name}_{args.pairs}_t{args.tlen}"
           + (f"_tr{args.target_tr:g}" if args.target_tr else "")
           + {"preserve": "_amp", "official": "_off", "zscore": ""}[args.amplitude]
           + ("" if args.same_protocol else "_xproto"))

    rows = []
    for (ptid, r1, r2, o1, o2, p1, p2) in keep:
        row = {"PTID": ptid, "FILE_NAME_T1": r1["FILE_NAME"], "FILE_NAME_T2": r2["FILE_NAME"],
               "month_T1": int(r1["month"]), "month_T2": int(r2["month"]),
               "interval_months": int(r2["month"] - r1["month"]),
               "protocol_group": p1, "protocol_group_T2": p2,
               "n_timepoints_raw_T1": int(o1[1]), "n_timepoints_raw_T2": int(o2[1]),
               "gain_T1": o1[2], "gain_T2": o2[2],
               "amplitude": args.amplitude, "scan_release": scan_dir.name}
        for k in have:
            col = OUTCOMES[k][0]
            v1, v2 = pd.to_numeric(r1[col], errors="coerce"), pd.to_numeric(r2[col], errors="coerce")
            row[f"{col}_T1"], row[f"{col}_T2"] = v1, v2
            # Sign convention matched to the RCTs: POSITIVE = improvement.
            row[f"d_{k}"] = v2 - v1
        for col in ("AGE", "PTGENDER", "PTEDUCAT", "APOE4", "DX", "SITE", "COLPROT"):
            if col in r1.index:
                row[col] = r1[col]
        rows.append(row)
    man = pd.DataFrame(rows)

    # `label` is what the fold assigner stratifies on; use the primary outcome's sign.
    primary = "mem" if "mem" in have else have[0]
    man["label"] = (man[f"d_{primary}"] > 0).astype("Int64")

    torch.save(x, out_dir / f"ADNI_x_{tag}.pt")
    for k in have:
        torch.save(torch.tensor(man[f"d_{k}"].values, dtype=torch.float32),
                   out_dir / f"ADNI_d{k}_{tag}.pt")
    man.to_csv(out_dir / f"ADNI_manifest_{tag}.csv", index=False)

    print(f"\nX={tuple(x.shape)} over {man['PTID'].nunique()} subjects "
          f"({len(man)} pairs, --pairs {args.pairs})")
    print(f"interval months: median {man.interval_months.median():.0f} "
          f"p10 {man.interval_months.quantile(.1):.0f} p90 {man.interval_months.quantile(.9):.0f}")
    for k in have:
        d = man[f"d_{k}"].dropna()
        print(f"  d_{k} ({OUTCOMES[k][0]}, + = improvement): n {len(d)} "
              f"mean {d.mean():+.3f} sd {d.std():.3f} improved {(d > 0).mean() * 100:.0f}%")
    print(f"protocol groups: {man.protocol_group.value_counts().to_dict()}")
    print(f"wrote {out_dir}/ADNI_{{x,d*,manifest}}_{tag}.*")


if __name__ == "__main__":
    main()
