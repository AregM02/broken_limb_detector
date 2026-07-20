# Broken Limb Detector

Minimal checker library for NatNet broken-limb detection.

```python
import pandas as pd

from src.detector import detect

df = pd.read_parquet("data/validation.parquet")
events = detect(df)
```

`events` contains one row per frame, boolean bone columns, a global `broken`
column, and a `checkers` list showing which checks raised anything in that
frame.

Use the notebook for validation and `src.utils.visualization` for plotting.

## Parameter Sweeps

Run the Absolute Limit Checker margin sweep from the project root:

```bash
python -m src.sweep.abs_limit_sweep data/validation.parquet
```

This uses the default margin values:

```text
-3 -2 -1 0 1 2 3
```

Custom margin values can be provided with `--margins`:

```bash
python -m src.sweep.abs_limit_sweep \
    data/validation.parquet \
    --margins -2 -1 0 1 2
```

Every combination of six margin values is tested for each bone. The script
prints the margins with the best F1 score for every bone.
