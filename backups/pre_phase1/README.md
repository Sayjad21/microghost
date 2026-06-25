# Pre-Phase-1 backups

Created before inference post-processing improvements (grid-cell blacklist,
wider corner filter, NMS 0.35).

| File | Description |
|------|-------------|
| `inference.py` | Decode + `filter_spurious_detections` before grid metadata |
| `config.py` | NMS 0.45, MAX_DETECTIONS 10, original corner filter only |

Restore:
```bash
cp backups/pre_phase1/inference.py inference.py
cp backups/pre_phase1/config.py config.py
```
