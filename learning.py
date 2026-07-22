import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.detector.checkers import (
    END_EFFECTOR_BONES,
    AbsoluteLimitChecker,
    GravityAlignmentChecker,
)
from src.skeletons.natnet_skeleton import NATNET


def features_for_bone(absolute, gravity, bone):
    bone_idx = NATNET.index(bone)
    quaternions = absolute["local_quaternions"][:, bone_idx].copy()
    quaternions[quaternions[:, 3] < 0] *= -1  # q and -q are equivalent

    return np.column_stack(
        [
            quaternions,
            absolute["axis_excess"][:, bone_idx],
            gravity["gravity"][:, bone_idx],
            gravity["cosine"][:, bone_idx],
            gravity["valid"][:, bone_idx],
            absolute["violated"][:, bone_idx],
            gravity["raw_violated"][:, bone_idx],
            gravity["violated"][:, bone_idx],
        ]
    )


if __name__ == "__main__":
    df = pd.read_parquet("data/validation.parquet")
    labels = df.filter(regex=r"^label_Broken").notna()

    absolute = AbsoluteLimitChecker(calibrate=True, tpose_window=500).diagnostics(df)
    gravity = GravityAlignmentChecker(
        calibration_window=500,
        temporal_filter=True,
    ).diagnostics(df)

    groups = np.arange(len(df)) // 500
    train, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(
            df,
            groups=groups,
        )
    )

    rows = []
    for bone in END_EFFECTOR_BONES:
        x = features_for_bone(absolute, gravity, bone)
        y = labels[f"label_Broken{bone}"].to_numpy(bool)

        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(x[train], y[train])
        prediction = model.predict(x[test])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y[test],
            prediction,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "bone": bone,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(y[test].sum()),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
