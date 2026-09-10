#!/usr/bin/env python3
"""
FP.6 Root-Cause Example Analysis

Generates the supporting figure (fp6_rootcause_analysis.png) that accompanies the four
keypoint-match overlay images (rootcause_rc*.png) illustrating the four root causes of the
large camera-TTC outliers:

  RC1 Keypoint distribution sensitivity  -> HARRIS/SIFT, frame 8  (TTC 73.6 s, 250 pairs)
  RC2 Insufficient keypoint pairs         -> HARRIS/BRIEF, frame 2 (TTC 94.0 s, 62 pairs)
  RC3 Descriptor mismatch                 -> ORB/FREAK, frame 14   (TTC 54.2 s, 28 pairs)
  RC4 Frame-specific keypoint availability -> HARRIS/BRIEF, frame 1 (NaN, 2 pairs)

The overlays are produced by the C++ pipeline (see showRootCauseKeypointOverlay in
camFusion_Student.cpp and the FP.6 root-cause hook in FinalProject_Camera.cpp). This
script reads the per-pair distance ratios exported by the same hook plus the main
combination TTC CSV to draw a 2x2 panel:
  - RC1: distance-ratio histogram (median near 1.0 -> huge TTC)
  - RC2: valid pair counts per detector at frame 2 (HARRIS vs the rest)
  - RC3: mean keypoint count for ORB by descriptor (FREAK drops 500 -> ~180)
  - RC4: valid pairs per frame for HARRIS/BRIEF (frame 1 below the 5-pair threshold -> NaN)

Usage:
    python fp6_rootcause_analysis.py
    python fp6_rootcause_analysis.py --csv output/ttc_camera_combinations.csv --output output
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TTC becomes NaN when fewer than this many valid distance-ratio pairs are available
MIN_PAIRS_FOR_TTC = 5


def load_combo(csv_path):
    return pd.read_csv(csv_path)


def load_ratios(output_dir, tag):
    path = os.path.join(output_dir, f"rootcause_ratios_{tag}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)["ratio"].to_numpy()


def plot_rc1(ax, ratios):
    """RC1: distance-ratio distribution with median near 1.0 -> huge TTC."""
    if ratios is None or len(ratios) == 0:
        ax.text(0.5, 0.5, "no ratio data", ha="center", va="center", transform=ax.transAxes)
        return
    med = float(np.median(ratios))
    ax.hist(ratios, bins=30, color="#4C72B0", edgecolor="white", alpha=0.9)
    ax.axvline(med, color="crimson", lw=2, ls="--", label=f"median = {med:.4f}")
    ax.axvline(1.0, color="black", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("distance ratio (curr / prev)")
    ax.set_ylabel("pair count")
    ax.set_title("RC1  HARRIS/SIFT, frame 8  (250 valid pairs)")
    ax.legend(fontsize=8)
    # TTC annotation
    dT = 1.0 / 10.0
    ttc = -dT / (1.0 - med)
    ax.text(0.98, 0.98, f"median ratio {med:.4f}  ->  TTC {abs(ttc):.1f} s",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))


def plot_rc2(ax, df):
    """RC2: valid pair counts per detector at frame 2 (HARRIS has very few)."""
    sub = df[df.frame_index == 2]
    order = (sub.groupby("detector")["num_pairs"].median()
             .sort_values(ascending=False).index.tolist())
    vals = [sub[sub.detector == d]["num_pairs"].median() for d in order]
    colors = ["crimson" if d == "HARRIS" else "#4C72B0" for d in order]
    ax.bar(order, vals, color=colors, edgecolor="white")
    ax.set_ylabel("median valid pairs (frame 2)")
    ax.set_title("RC2  Valid pairs per detector, frame 2")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax.text(0.5, 0.97, "HARRIS/BRIEF: 62 pairs -> TTC 94.0 s\nSHITOMASI/BRIEF: 547 pairs -> TTC 14.2 s",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))


def plot_rc3(ax, df):
    """RC3: ORB keypoint count by descriptor - FREAK cannot describe most ORB keypoints."""
    orb = df[df.detector == "ORB"].groupby("descriptor")["keypoint_count"].mean()
    order = orb.sort_values(ascending=False).index.tolist()
    vals = [orb[d] for d in order]
    colors = ["crimson" if d == "FREAK" else "#4C72B0" for d in order]
    ax.bar(order, vals, color=colors, edgecolor="white")
    ax.set_ylabel("mean keypoint count (ORB detector)")
    ax.set_title("RC3  ORB keypoints surviving description")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(vals) * 1.18)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax.text(0.5, 0.97, "FREAK drops ORB keypoints 500 -> ~180\n-> few matches, unstable TTC",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))


def plot_rc4(ax, df):
    """RC4: HARRIS/BRIEF valid pairs per frame - frame 1 below 5-pair threshold -> NaN."""
    sub = df[(df.detector == "HARRIS") & (df.descriptor == "BRIEF")].sort_values("frame_index")
    frames = sub.frame_index.to_numpy()
    pairs = sub.num_pairs.to_numpy()
    valid = ~sub.ttc_camera.isna().to_numpy()
    ax.bar(frames[valid], pairs[valid], color="#4C72B0", edgecolor="white", label="valid TTC")
    ax.bar(frames[~valid], pairs[~valid], color="crimson", edgecolor="white", label="NaN (TTC)")
    ax.axhline(MIN_PAIRS_FOR_TTC, color="black", ls="--", lw=1, label=f"{MIN_PAIRS_FOR_TTC}-pair threshold")
    ax.set_xlabel("frame index")
    ax.set_ylabel("valid distance-ratio pairs")
    ax.set_title("RC4  HARRIS/BRIEF pairs per frame")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(1, pairs[list(frames).index(1)] + 1, "2 pairs -> NaN", fontsize=8, color="crimson")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def main():
    parser = argparse.ArgumentParser(description="FP.6 root-cause example analysis")
    parser.add_argument("--csv", default="output/ttc_camera_combinations.csv",
                        help="Path to ttc_camera_combinations.csv")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    df = load_combo(args.csv)
    rc1_ratios = load_ratios(args.output, "rc1_harris_sift_f8")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("FP.6 Root-Cause Examples for Large Camera-TTC Outliers",
                 fontsize=14, fontweight="bold")
    plot_rc1(axes[0, 0], rc1_ratios)
    plot_rc2(axes[0, 1], df)
    plot_rc3(axes[1, 0], df)
    plot_rc4(axes[1, 1], df)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(args.output, "fp6_rootcause_analysis.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
