"""Metric helpers for bone-wise detector evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def bone_wise_metrics(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    bones: Sequence[str],
    *,
    zero_division: int | float = 0,
) -> pd.DataFrame:
    """
    Thin wrapper around ``sklearn.metrics.precision_recall_fscore_support``.
    Return bone-wise precision, recall, F1, and support.
    
    Parameters
    ----------
    labels:
        A dataframe containing binary broken/ok labels for all bones.
    predictions:
        Prediction result from the detector. 
    bones:
        A list of bone names for which metrics should be calculated. 
    """

    labels = labels.rename(columns=lambda col: col.removeprefix("label_"))
    rows = []
    for bone in bones:
        col = f"Broken{bone}"
        y_true = labels[col].to_numpy(bool)
        y_pred = predictions[col].to_numpy(bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=zero_division,
        )
        rows.append(
            {
                "bone": bone,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(y_true.sum()),
            }
        )

    return pd.DataFrame(rows)
