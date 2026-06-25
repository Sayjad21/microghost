# Paste into config.py before Phase-2 (3-anchor) retrain.
# Requires fresh training — old 2-anchor checkpoint will not load.

NUM_ANCHORS = 3
DEFAULT_ANCHOR_RATIOS = [1.6, 2.5, 3.5]
DEFAULT_ANCHOR_SIZES = [0.095, 0.127, 0.145]
CONFIDENCE_THRESHOLD = 0.12
