"""Sweep gravity cosine and temporal-filter parameters."""

import argparse
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from ..detector.checkers import END_EFFECTOR_BONES, GravityAlignmentChecker
from ..skeletons.natnet_skeleton import NATNET


DEFAULT_THRESHOLDS = np.linspace(0.5, 0.99, 100)
DEFAULT_WINDOWS = (10, 20, 30, 40)


def search_bone_thresholds(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    thresholds: list[float],
    windows: list[int],
    bone: str,
) -> pd.DataFrame:
    checker = GravityAlignmentChecker(
        bones=[bone],
        calibration_window=500,
        temporal_filter=True,
    )
    bone_idx = NATNET.index(bone)
    y_true = labels[f"label_Broken{bone}"].to_numpy(bool)
    rows = []
    best_f1 = 0.0
    configurations = [
        (threshold, window, required)
        for threshold in thresholds
        for window in windows
        for required in range(1, window + 1)
    ]

    progress = tqdm(
        configurations,
        desc=f"Optimizing {bone}",
        unit="config",
        position=END_EFFECTOR_BONES.index(bone),
        leave=True,
        dynamic_ncols=True,
    )

    for threshold, window, required in progress:
        checker.threshold_cos = threshold
        checker.temporal_window = window
        checker.temporal_required = required
        y_pred = checker.check(df)[:, bone_idx]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            progress.set_postfix(
                best_f1=f"{f1:.3f}",
                precision=f"{precision:.3f}",
                recall=f"{recall:.3f}",
                refresh=False,
            )

        rows.append(
            {
                "bone": bone,
                "threshold_cos": threshold,
                "threshold_deg": np.degrees(np.arccos(threshold)),
                "temporal_window": window,
                "temporal_required": required,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    args = parser.parse_args()

    if any(not -1 <= threshold <= 1 for threshold in args.thresholds):
        parser.error("thresholds must be between -1 and 1")
    if any(window < 1 for window in args.windows):
        parser.error("windows must be positive")

    df = pd.read_parquet(args.filename)
    labels = df.filter(regex=r"^label_Broken").notna()
    search = partial(
        search_bone_thresholds,
        df,
        labels,
        args.thresholds,
        args.windows,
    )

    with Pool(processes=len(END_EFFECTOR_BONES)) as pool:
        results = pd.concat(
            pool.map(search, END_EFFECTOR_BONES),
            ignore_index=True,
        )

    best = (
        results.sort_values(
            ["bone", "f1", "precision", "temporal_window"],
            ascending=[True, False, False, True],
        )
        .groupby("bone", as_index=False)
        .first()
    )

    print(
        best[
            [
                "bone",
                "threshold_cos",
                "threshold_deg",
                "temporal_window",
                "temporal_required",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )
