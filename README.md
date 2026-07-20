Broken Limb Detector
====================

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

Use the notebook for validation, and plotting with `src.utils.visualization`.
