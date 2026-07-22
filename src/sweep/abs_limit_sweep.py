"""Does a margin sweep for the limits of the absolute limit checker."""

import argparse
from functools import partial
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from ..detector.checkers import ABSOLUTE_LIMITS, AbsoluteLimitChecker
from ..skeletons.natnet_skeleton import NATNET


MARGIN_VALUES = (-4, -3, -2., 2., 3., 4.)
MARGIN_COLUMNS = (
    "lower_x",
    "upper_x",
    "lower_y",
    "upper_y",
    "lower_z",
    "upper_z",
)


def search_bone_margins(
    df: pd.DataFrame,
    labels: pd.DataFrame,
    margins: list[float],
    bone: str,
) -> pd.DataFrame:
    checker = AbsoluteLimitChecker(
        limits={bone: ABSOLUTE_LIMITS[bone]},
        calibrate=True,
        tpose_window=500,
    )

    bone_idx = NATNET.index(bone)
    y_true = labels[f"label_Broken{bone}"].to_numpy(bool)
    rows = []
    best_f1 = 0.0

    progress = tqdm(
        product(margins, repeat=6),
        total=len(margins) ** 6,
        desc=f"Optimizing {bone}",
        unit="config",
        position=list(ABSOLUTE_LIMITS).index(bone),
        leave=True,
        dynamic_ncols=True,
    )

    for values in progress:
        checker.margins = {bone: np.asarray(values).reshape(3, 2)}
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
                **dict(zip(MARGIN_COLUMNS, values)),
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
        "--margins",
        type=float,
        nargs="+",
        default=MARGIN_VALUES,
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.filename)
    labels = df.filter(regex=r"^label_Broken").notna()

    search = partial(search_bone_margins, df, labels, args.margins)

    with Pool(processes=len(ABSOLUTE_LIMITS)) as pool:
        results = pd.concat(
            pool.map(search, ABSOLUTE_LIMITS),
            ignore_index=True,
        )

    results["total_margin"] = results[list(MARGIN_COLUMNS)].sum(axis=1)

    best = (
        results.sort_values(
            ["bone", "f1", "total_margin"],
            ascending=[True, False, True],
        )
        .groupby("bone", as_index=False)
        .first()
    )

    print(
        best[
            ["bone", *MARGIN_COLUMNS, "precision", "recall", "f1"]
        ].to_string(index=False)
    )