Broken Limb Detector
====================

Use `main.py` as the single CLI entry point:

```bash
python main.py detect parquets/test_rhand.processed.parquet --no-plot
python main.py skeleton parquets/test_rhand.processed.parquet --frame 1000
python main.py cosine parquets/test_rhand.processed.parquet --bone RHand --threshold 0.8
```

Commands:

- `detect`: run the RPY limit checker and gravity alignment checker.
- `skeleton`: plot the NatNet skeleton with gravity overlays.
- `cosine`: plot gravity-fit cosine similarity over time.
