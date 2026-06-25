# Pre A+B training backups

Before Phase 3 training improvements (neighbor positive cells + focal objectness).

**Fixed (current code):** neighbor cells get soft objectness (0.5) only; bbox + CIoU
loss only on center cells (1.0). The first Phase 3 run wrongly assigned full bbox to
all 9 cells in the 3×3 neighborhood.

Restore old broken encoder:
```bash
cp backups/pre_ab/preprocessing.py preprocessing.py
cp backups/pre_ab/training.py training.py
cp backups/pre_ab/config.py config.py
```

Phase 1 checkpoint (best 2-anchor model):
`checkpoints/best_phase1_baseline.pth`
