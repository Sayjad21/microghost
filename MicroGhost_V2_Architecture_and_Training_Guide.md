
# MicroGhost-V2: Asynchronous Dual-Branch Architecture  
## Complete Architecture Design, Training Pipeline & Dataset Strategy  
*Designed for ESP32-S3 · Jungle Terrain · Camouflaged Intruder Detection*

---

## Table of Contents

1. [Why We Are Rebuilding — The Core Problem](#1-why-we-are-rebuilding)
2. [The New Design Philosophy](#2-the-new-design-philosophy)
3. [New Architecture: MicroGhost-V2](#3-new-architecture-microghost-v2)
4. [What Replaces FPN — The BiFusion Neck](#4-what-replaces-fpn--the-bifusion-neck)
5. [The Full Model: Layer-by-Layer Specification](#5-the-full-model-layer-by-layer-specification)
6. [Dataset Strategy: What You Have and How to Use It] 7. [The 4-Phase Training Pipeline](#7-the-4-phase-training-pipeline)
8. [Loss Functions: What Changes](#8-loss-functions-what-changes)
9. [Config Changes Checklist](#9-config-changes-checklist)
10. [Implementation Priorities](#10-implementation-priorities)

---

## 1. Why We Are Rebuilding

Your original MicroGhost-Thermal had **two fatal architectural decisions** that caused all three major error categories:

### Problem A: Early Fusion = Single Point of Failure (the "broken camera" problem)

```
RGB Stem  ──┐
             ├── cat() ──► Single fused backbone ──► FPN ──► Heads
Thm Stem  ──┘
```

When you `torch.cat()` at the stem level and then pass through a **single shared backbone**, the two modalities are permanently entangled by Scale 1. This means:

- If the thermal camera fails at night → the whole backbone gets garbage input on 16 of 32 channels → detection collapses completely
- If the RGB camera is dark (nighttime jungle) → same collapse on 16 of 32 channels
- Neither branch can "veto" the other because they are the same tensor after Scale 1

This is precisely why you got hot car bonnets firing: the thermal branch screams "hot blob" and the RGB branch has no veto mechanism — they are both just channels in the same tensor from the moment they are concatenated.

### Problem B: FPN is the Wrong Tool for Your Scenario

FPN was designed for ImageNet-scale detection where objects span many different sizes at the same time in the same image (a bus next to a person next to a dog). Your scenario is different:

- One or two humans at roughly similar distances
- Ground-level camera with constrained field of view
- Extreme background clutter from foliage
- Target occupies a **predictable range of sizes** relative to your camera mount height

FPN's upsample-and-add operation also **destroys the per-modality independence** you need, because by the time features reach the FPN, both modalities have been completely merged. FPN is also symmetric in its treatment of all spatial locations — it has no mechanism to suppress known-background regions (hot leaves, hot engine parts, tree trunks).

**What you need instead**: a neck that aggregates features asymmetrically, with learned spatial importance weights that suppress clutter regions. That is the **BiFusion Neck with Deformable Aggregation**.

---

## 2. The New Design Philosophy

MicroGhost-V2 is built on three principles:

### Principle 1: Parallel Branches, Late Gated Fusion

Both RGB and Thermal backbones run **completely independently** all the way to the neck. They share no weights and no activations until the fusion gate. This means:

- Thermal camera fails → RGB branch continues running, detection degrades gracefully but does not collapse
- RGB is dark (nighttime) → Thermal branch continues with full confidence, gate automatically mutes the RGB evidence
- No single point of failure

### Principle 2: The Gate Decides Who Speaks

A tiny **Energy Gate module** at the neck computes, per spatial location, how much confidence to assign to each branch. At a hot car bonnet region, the RGB branch features will show metal/car texture — the gate learns that this RGB pattern should suppress the thermal activation at that location. This is the RGB veto mechanism that was completely missing in V1.

### Principle 3: Train Each Branch to Survive Alone (CMM)

During training, we randomly force either the RGB or the thermal input to zero/noise, ensuring each branch learns to produce useful features independently. This is Causal Mode Multiplexer (CMM, CVPR 2024). It directly solves the "inconsistent hot bonnet frames" problem by eliminating the statistical shortcut "high thermal activation → person".

---

## 3. New Architecture: MicroGhost-V2

### High-Level Flow

```
RGB Input                         Thermal Input
(3, 160, 128)                     (1, 160, 128)
     │                                  │
     ▼                                  ▼
 RGB Ghost Stem                  Thm Ghost Stem
 (16, 80, 64)                    (16, 80, 64)
     │                                  │
     ▼                                  ▼
 RGB Scale 1                     Thm Scale 1
 GhostBottleneck                 GhostBottleneck
 (16, 40, 32)                    (16, 40, 32)
     │                                  │
     ▼                                  ▼
 RGB Scale 2                     Thm Scale 2
 InvertedResidual                InvertedResidual
 (12, 20, 16)  ─────────────────► Energy Gate ◄── (12, 20, 16)
                                       │
                                       ▼
                                 Gated Fused (12, 20, 16)
                                       │
                                       ▼
 RGB Scale 3                     BiFusion Neck
 InvertedResidual     ─────────► (receives S3 from both branches)
 (8, 10, 8)                           │
                    ┌─────────────────┼─────────────────┐
                    ▼                                   ▼
               P2 (12, 20, 16)                   P3 (12, 10, 8)
               Small Head                         Large Head
                    │                                   │
                    └──────────────┬────────────────────┘
                                   ▼
                          Reliability Classifier
                          (Person_Visible /
                           Person_Camouflaged /
                           Background)
```

### What Changed vs V1

| Component | V1 | V2 |
|---|---|---|
| Fusion point | Stem output (immediately) | Scale 2 output (late) |
| Fusion method | `torch.cat()` — equal weight always | Energy Gate — learned per-location weights |
| Shared backbone after fusion | Yes (Scale 1, 2, 3 all shared) | No — separate Scale 1, 2, 3 per branch |
| Camera failure behavior | Full collapse | Graceful degradation |
| Neck | Lightweight FPN (upsample + add) | BiFusion Neck (deformable aggregation) |
| FPN spatial uniformity | All locations treated equally | Learned spatial importance per location |
| Training strategy | Single modality joint input always | CMM: random single-modality dropout |
| Anchor count | 2 per cell | 3 per cell |
| Log clamp | ±3 | ±4.5 |
| NMS threshold | ~0.5 | 0.35 |

---

## 4. What Replaces FPN — The BiFusion Neck

### Why FPN Fails Here

Standard FPN does this at each pyramid level:

```
P[i] = smooth_conv( lateral(C[i]) + upsample(P[i+1]) )
```

The `upsample(P[i+1])` is a **bilinear interpolation** — every upsampled cell contributes equally to every location. A hot-leaf patch at location (10, 5) gets the same treatment as the actual human at location (10, 8). The neck has no concept of "this background region is noisy clutter, suppress it."

### The BiFusion Neck (Replaces FPN)

BiFusion is inspired by BiFPN (EfficientDet, CVPR 2020) but stripped down for your parameter budget. The key difference from FPN:

**FPN:** `P_out = Conv(C_shallow + upsample(C_deep))`

**BiFusion:** `P_out = Conv(w1*C_shallow + w2*upsample(C_deep))` where `w1, w2` are **learned, normalized, per-channel scalar weights** — not fixed 1.0.

This means the network can learn: "at this scale, shallow spatial detail from RGB is more important than the deep semantic from thermal" or vice versa. The weights are learned end-to-end during training.

Additionally, BiFusion adds a **bottom-up refinement pass** after the top-down pass, so information flows in both directions. This recovers fine spatial detail that FPN loses in the upsample step.

```
Top-down pass (like FPN):
  P3_td = BN(w1*C_s3 + eps) / (w1 + w2 + eps) * upsample(C_deep)
                [semantic context flows down]

Bottom-up pass (NEW — FPN doesn't have this):
  P2_out = BN(w3*C_s2 + w4*P2_td + w5*maxpool(P3_td))
                [spatial detail flows back up]
```

For your parameter budget, BiFusion costs approximately the same as your current FPN — the scalar weights `w1, w2` are nearly free. The bottom-up pass adds one `maxpool + BN` per level.

### The Energy Gate (Replaces simple concatenation)

At the Scale 2 output, before the features enter the BiFusion Neck, the Energy Gate runs:

```python
# Per-channel L2 energy of each branch's feature map
E_rgb  = feat_rgb_s2.pow(2).mean(dim=1, keepdim=True)   # (B, 1, H, W)
E_thm  = feat_thm_s2.pow(2).mean(dim=1, keepdim=True)   # (B, 1, H, W)

# Softmax across the two branches (per spatial location)
weights = torch.softmax(torch.stack([E_rgb, E_thm], dim=1), dim=1)
w_rgb, w_thm = weights[:, 0], weights[:, 1]              # each (B, 1, H, W)

# Weighted combination (NOT concatenation)
fused_s2 = w_rgb * feat_rgb_s2 + w_thm * feat_thm_s2    # (B, 12, H, W)
```

At a hot car bonnet location:
- `feat_rgb_s2` has car-texture features → `E_rgb` is **high** (many strong activations, but they encode "car", not "human")
- `feat_thm_s2` has high activation → `E_thm` is also high
- But the gate is **learned** — after training with CMM and contrast loss, the network learns that when RGB energy encodes "metallic flat surface" but thermal is high, this is a false alarm pattern → `w_rgb` for that cell gets large → it suppresses the thermal activation

The Energy Gate adds approximately 2 × `1×1 Conv(12→1)` per branch = ~30 parameters total.

---

## 5. The Full Model: Layer-by-Layer Specification

### 5.1 RGB Branch

```
Input: (B, 3, 160, 128)

rgb_stem:
  Conv2d(3, 8, 3, stride=2, pad=1)       → (B, 8, 80, 64)
  BatchNorm2d(8) + ReLU6
  GhostModule(8, 16, kernel=1, stride=1)  → (B, 16, 80, 64)

rgb_scale1:
  GhostBottleneck(16, 32, 16, stride=2)   → (B, 16, 40, 32)
  GhostBottleneck(16, 32, 16, stride=1)   → (B, 16, 40, 32)

rgb_scale2:
  InvertedResidual(16, 12, stride=2, expand=6)  → (B, 12, 20, 16)
  InvertedResidual(12, 12, stride=1, expand=6)  → (B, 12, 20, 16)
  ──► feat_rgb_s2  [goes to Energy Gate]

rgb_scale3:
  InvertedResidual(12, 8, stride=2, expand=6)   → (B, 8, 10, 8)
  InvertedResidual(8,  8, stride=1, expand=6)   → (B, 8, 10, 8)
  ──► feat_rgb_s3  [goes to BiFusion Neck]
```

### 5.2 Thermal Branch

```
Input: (B, 1, 160, 128)

thm_stem:
  Conv2d(1, 8, 3, stride=2, pad=1)        → (B, 8, 80, 64)
  BatchNorm2d(8) + ReLU6
  GhostModule(8, 16, kernel=1, stride=1)  → (B, 16, 80, 64)

thm_scale1:
  GhostBottleneck(16, 32, 16, stride=2)   → (B, 16, 40, 32)
  GhostBottleneck(16, 32, 16, stride=1)   → (B, 16, 40, 32)

thm_scale2:
  InvertedResidual(16, 12, stride=2, expand=6)  → (B, 12, 20, 16)
  InvertedResidual(12, 12, stride=1, expand=6)  → (B, 12, 20, 16)
  ──► feat_thm_s2  [goes to Energy Gate]

thm_scale3:
  InvertedResidual(12, 8, stride=2, expand=6)   → (B, 8, 10, 8)
  InvertedResidual(8,  8, stride=1, expand=6)   → (B, 8, 10, 8)
  ──► feat_thm_s3  [goes to BiFusion Neck]
```

> **Note:** Both branches have **identical architecture** but **completely separate weights**. They do not share any parameters. This is the key change that enables graceful camera failure.

### 5.3 Energy Gate (at Scale 2 output)

```python
class EnergyGate(nn.Module):
    """
    Per-location gating between RGB and Thermal at Scale 2.
    Lets the network suppress whichever branch has unreliable content.
    ~30 parameters. Applied BEFORE BiFusion Neck.
    """
    def __init__(self, channels):
        super().__init__()
        # Learned energy projection (refines raw L2 energy)
        self.proj_rgb = nn.Conv2d(channels, 1, 1, bias=True)
        self.proj_thm = nn.Conv2d(channels, 1, 1, bias=True)

    def forward(self, feat_rgb, feat_thm):
        e_rgb = self.proj_rgb(feat_rgb)          # (B, 1, H, W)
        e_thm = self.proj_thm(feat_thm)          # (B, 1, H, W)
        weights = torch.softmax(
            torch.stack([e_rgb, e_thm], dim=1),  # (B, 2, 1, H, W)
            dim=1
        )
        w_rgb = weights[:, 0]                    # (B, 1, H, W)
        w_thm = weights[:, 1]                    # (B, 1, H, W)
        fused = w_rgb * feat_rgb + w_thm * feat_thm
        return fused, w_rgb, w_thm               # return weights for loss
```

The returned `w_rgb, w_thm` are used in the auxiliary contrast loss during training (see Section 8). At inference export, you can fold them in.

### 5.4 BiFusion Neck (Replaces LightweightFPN)

```python
class BiFusionNeck(nn.Module):
    """
    Bidirectional weighted feature pyramid.
    Receives S2 and S3 features from BOTH branches,
    plus the gated fused_s2 from EnergyGate.

    Inputs:
        fused_s2:  (B, 12, 20, 16) — gated output from EnergyGate
        feat_rgb_s3: (B, 8, 10, 8)
        feat_thm_s3: (B, 8, 10, 8)

    Outputs:
        p2: (B, FPN_CH, 20, 16) — small/distant target features
        p3: (B, FPN_CH, 10, 8)  — large/close target features
    """
    def __init__(self, s2_ch=12, s3_ch=8, out_ch=12):
        super().__init__()
        FPN_CH = out_ch

        # Lateral projections to unified channel count
        self.lat_s2  = nn.Conv2d(s2_ch, FPN_CH, 1, bias=False)
        self.lat_rgb = nn.Conv2d(s3_ch, FPN_CH, 1, bias=False)
        self.lat_thm = nn.Conv2d(s3_ch, FPN_CH, 1, bias=False)

        # Learned BiFPN weights (log-space, softmax-normalized)
        # Top-down: P3 = w1*rgb_s3 + w2*thm_s3
        self.w_td = nn.Parameter(torch.ones(2))

        # Bottom-up P2: w3*s2 + w4*P3_upsampled
        self.w_bu = nn.Parameter(torch.ones(2))

        # Refinement convolutions (DW-separable)
        self.refine_p3 = self._dw_sep(FPN_CH, FPN_CH)
        self.refine_p2 = self._dw_sep(FPN_CH, FPN_CH)

    def _dw_sep(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, fused_s2, feat_rgb_s3, feat_thm_s3):
        eps = 1e-4

        # Project to unified channels
        lat_s2  = self.lat_s2(fused_s2)         # (B, CH, 20, 16)
        lat_rgb = self.lat_rgb(feat_rgb_s3)      # (B, CH, 10, 8)
        lat_thm = self.lat_thm(feat_thm_s3)      # (B, CH, 10, 8)

        # ── Top-down: fuse S3 from both branches ──────────────────────
        w_td = F.relu(self.w_td)
        w_td = w_td / (w_td.sum() + eps)
        p3_td = self.refine_p3(
            w_td[0] * lat_rgb + w_td[1] * lat_thm
        )                                        # (B, CH, 10, 8)

        # ── Bottom-up: upsample P3 and merge with S2 ──────────────────
        p3_up = F.interpolate(p3_td, size=lat_s2.shape[2:], mode='nearest')
        w_bu = F.relu(self.w_bu)
        w_bu = w_bu / (w_bu.sum() + eps)
        p2_out = self.refine_p2(
            w_bu[0] * lat_s2 + w_bu[1] * p3_up
        )                                        # (B, CH, 20, 16)

        return p2_out, p3_td
```

### 5.5 Detection Heads (Upgraded to 3 Anchors)

No structural change from V1 SSDLiteHead, but:
- `NUM_ANCHORS = 3` (was 2)
- Anchor 3 is tuned for two-person-wide bounding boxes (h/w ratio ≈ 1.2)
- Log clamp changed from `clamp(-3, 3)` → `clamp(-4.5, 4.5)` everywhere in training.py, preprocessing.py, and inference.py

### 5.6 Reliability Classifier (Replaces IntrusionClassifier)

The classifier is functionally identical to V1 but receives `w_rgb` and `w_thm` from the EnergyGate as auxiliary input. This lets it know at inference time which modality was dominant — directly enabling the Visible vs Camouflaged classification:

- If `w_thm >> w_rgb` at the detection region → Thermal dominant → likely `Person_Camouflaged` (ghillie suit, dark clothing)
- If `w_rgb ≈ w_thm` → both modalities agree → `Person_Visible`

```python
class ReliabilityClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=4, hidden_dim=32):
        super().__init__()
        # +2 for the gate weight scalars (w_rgb_avg, w_thm_avg)
        self.modality_gate = nn.Sequential(
            nn.Linear(in_channels * 2 + 2, hidden_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_dim // 2, in_channels * 2),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_p2, feat_p3, obj_p2, obj_p3, w_rgb, w_thm):
        # Same objectness-weighted pooling as V1
        attn_p2 = torch.sigmoid(obj_p2.max(dim=1, keepdim=True)[0])
        attn_p3 = torch.sigmoid(obj_p3.max(dim=1, keepdim=True)[0])
        p2_w = (feat_p2 * attn_p2).sum([2,3]) / (attn_p2.sum([2,3]) + 1e-6)
        p3_w = (feat_p3 * attn_p3).sum([2,3]) / (attn_p3.sum([2,3]) + 1e-6)
        combined = torch.cat([p2_w, p3_w], dim=1)   # (B, 2C)

        # Append gate summary scalars
        gate_summary = torch.cat([
            w_rgb.mean(dim=[1,2,3]),   # (B,)
            w_thm.mean(dim=[1,2,3]),   # (B,)
        ], dim=1).unsqueeze(1)         # (B, 2)
        gate_summary = gate_summary.squeeze(1)       # (B, 2)

        gate_input = torch.cat([combined, gate_summary], dim=1)
        attn_weights = self.modality_gate(gate_input)
        attended = combined * attn_weights
        return self.classifier(attended)
```

### 5.7 Parameter Budget (Estimated)

| Component | V1 Params | V2 Params | Change |
|---|---|---|---|
| RGB Stem | ~1,200 | ~1,200 | same |
| Thm Stem | ~500 | ~500 | same |
| Scale 1 (shared in V1, dual in V2) | ~3,000 | ~6,000 | +3,000 |
| Scale 2 (shared in V1, dual in V2) | ~1,800 | ~3,600 | +1,800 |
| Scale 3 (shared in V1, dual in V2) | ~900 | ~1,800 | +900 |
| FPN / BiFusion Neck | ~2,400 | ~2,600 | +200 |
| Energy Gate | 0 | ~30 | +30 |
| Detection Heads (3 anchors) | ~800 | ~1,200 | +400 |
| Classifier | ~2,000 | ~2,100 | +100 |
| **TOTAL** | **~12,600** | **~19,030** | **+6,430** |

The added ~6.4K parameters doubles the backbone (dual branch vs single shared), which is the correct trade-off for fault tolerance and camera-veto capability. At INT8, 19K parameters = ~19KB — still comfortably within ESP32-S3 budget.

---

## 6. Dataset Strategy

### 6.1 What You Have and What It Teaches

| Dataset | Synchronized RGB+T | What It Teaches | Phase |
|---|---|---|---|
| LLVIP | ✅ 15,488 pairs | Human body shape in thermal and RGB, low-light | 1 |
| WiSARD paired subset | ✅ ~15,453 pairs | Wild terrain, foliage occlusion, all weather | 2 |
| ForestPersons (RGB) | ❌ RGB ONLY | Human silhouettes through tree canopy, ground-level perspective | 2 (RGB branch only) |
| ForestPersons IR | ❌ Thermal ONLY | Human thermal through canopy, ground-level perspective | 2 (thermal branch only) |
| MCOD | ✅ Paired | Camouflaged objects vs natural backgrounds, multispectral | 3 |
| Camo-M3FD | ✅ Paired | Foreground-background similarity in visible+thermal | 3 |

### 6.2 The Critical Insight About ForestPersons

ForestPersons RGB and ForestPersons IR are **not synchronized** (confirmed from dataset card). They were collected independently. However, you can still use them because of the **dual-branch architecture**:

In V2, each branch is an independent backbone. During training, you can run batches where:
- **Synchronized pairs** (LLVIP, WiSARD): both branches receive valid input
- **RGB-only batches** (ForestPersons RGB): only the RGB branch gets real data; thermal input is zeroed (CMM-RXTO mode)
- **Thermal-only batches** (ForestPersons IR): only the thermal branch gets real data; RGB input is zeroed (CMM-ROTX mode)

This is not a workaround — it is exactly what CMM training requires. ForestPersons becomes your primary CMM training data. You force the model to detect humans from a ground-level forest perspective using each branch in isolation.

This is why the dual-branch architecture makes ForestPersons usable even without synchronization.

---

## 7. The 4-Phase Training Pipeline

### Phase 1 — Human Shape Foundation (LLVIP Only)

**Goal:** Teach the dual-branch architecture what a human looks like in both thermal and RGB before it sees any jungle.

**Why LLVIP first:** The jungle datasets have humans at <5% of image area, partially occluded, and against complex backgrounds. If you train on these first from random initialization, the model cannot converge — it has no idea what it is looking for. LLVIP is clean, high-contrast, well-annotated, and teaches the strong human prior.

```
Dataset:    LLVIP (15,488 synchronized pairs)
Mode:       ROTO only (both branches receive real data)
Epochs:     40-60
LR:         1e-3, CosineAnnealing
Batch size: 32
Freeze:     Nothing — train all weights from random init
CMM:        OFF (both branches need to learn shape simultaneously)
Loss:       L_bbox + L_obj + L_cls (standard, no contrast loss yet)
Anchor:     K-Means on LLVIP labels — 3 anchors
```

**Checkpoint at:** best validation mAP@0.5 on LLVIP val split

---

### Phase 2 — Jungle Domain Transfer (WiSARD + ForestPersons)

**Goal:** Shift from urban street to forest floor. Teach foliage occlusion, dappled lighting, partial human silhouettes, ground-level perspective.

**Critical:** This is where you start CMM. Start with α=0.3 (only 30% of batches are single-modality) and increase to α=0.5 by epoch 20.

```
Dataset:    WiSARD paired subset (synchronized RGB+T)
            ForestPersons RGB (CMM-RXTO: thermal zeroed)
            ForestPersons IR (CMM-ROTX: RGB zeroed)
            LLVIP (keep 20% of batches — prevents catastrophic forgetting)

Sampling:   60% WiSARD  |  15% ForestPersons RGB  |
            15% ForestPersons IR  |  10% LLVIP

Mode:       Mixed — ROTO for WiSARD/LLVIP, RXTO for FP-RGB, ROTX for FP-IR
Epochs:     50-70
LR:         2e-4 (5× lower than Phase 1), CosineAnnealing
Batch size: 32
Freeze:     Nothing (full fine-tune, domain shift requires it)
CMM α:      0.3 → 0.5 over training
Loss:       L_bbox + L_obj + L_cls + 0.1 * L_contrast (activate contrast loss now)
```

**CMM loss formulation:**
```python
def cmm_loss(model, batch_rgb, batch_thm, labels, alpha=0.5):
    r = random.random()
    if r < 0.33:
        # ROTO — both modalities
        return compute_loss(model, batch_rgb, batch_thm, labels)
    elif r < 0.66:
        # RXTO — thermal only (RGB zeroed)
        return alpha * compute_loss(model, torch.zeros_like(batch_rgb), batch_thm, labels)
    else:
        # ROTX — RGB only (thermal zeroed)
        return alpha * compute_loss(model, batch_rgb, torch.zeros_like(batch_thm), labels)
```

**Checkpoint at:** best validation mAP@0.5 on WiSARD val split

---

### Phase 3 — Camouflage Fine-Tuning (MCOD + Camo-M3FD)

**Goal:** Teach the EnergyGate and Classifier to detect foreground-background similarity as a signal for camouflage, not as a reason to suppress detection.

**Freeze strategy:** Freeze `rgb_stem`, `thm_stem`, `rgb_scale1`, `thm_scale1`. Only update Scale 2, Scale 3, EnergyGate, BiFusion Neck, Heads, Classifier. This preserves the low-level feature detectors from Phases 1 and 2 while fine-tuning the high-level fusion logic.

```
Dataset:    MCOD (multispectral camouflaged, all scenes)
            Camo-M3FD (camouflaged-similarity RGB+T pairs)
            WiSARD (10% — prevents forgetting)

Sampling:   50% MCOD  |  40% Camo-M3FD  |  10% WiSARD

Epochs:     30-40
LR:         5e-5 (very low — fine-tuning only)
Batch size: 16 (smaller for small dataset)
Freeze:     rgb_stem, thm_stem, rgb_scale1, thm_scale1
CMM α:      0.5 (maintain from Phase 2)
Loss:       L_bbox + L_obj + L_cls + 0.15 * L_contrast
            Add L_gate_regularizer (see Section 8)
```

**Checkpoint at:** best combined mAP across MCOD + WiSARD val splits

---

### Phase 4 — Unfreeze and Polish

**Goal:** Final end-to-end tuning with everything unlocked, very low learning rate, full dataset mix.

```
Dataset:    Full mix: LLVIP (10%) + WiSARD (45%) +
            ForestPersons RGB+IR as CMM (30%) + MCOD+Camo-M3FD (15%)

Epochs:     20-30
LR:         1e-5 (very low)
Batch size: 32
Freeze:     Nothing
CMM α:      0.5
Loss:       Full loss with all terms
```

This is the polishing phase. If you added your own self-collected data from the deployment tree, **it goes here at 10% sampling rate alongside the existing mix**.

---

## 8. Loss Functions: What Changes

### Keep From V1

- `CIoU bbox loss` — keep exactly as-is
- `BCE objectness loss` — keep exactly as-is
- `CrossEntropy class loss` — keep exactly as-is

### Fix in V1 Code

In **both** `training.py` and `preprocessing.py`, change every instance of:
```python
.clamp(-3, 3)
```
to:
```python
.clamp(-4.5, 4.5)
```
This fixes tall-person bounding box saturation (MIS-2).

### Add: Contrast Loss (TFDet-style, training only)

Attach a 1×1 Conv auxiliary segmentation head to `p2` during training. Remove it at export.

```python
class AuxSegHead(nn.Module):
    """Training-only. Removed at export. Zero deployment cost."""
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, 1)

    def forward(self, feat_p2):
        return self.proj(feat_p2)   # (B, 1, H, W) — person/background logit

def contrast_loss(feat_p2, gt_mask, lambda_c=0.1):
    """
    gt_mask: (B, 1, H, W) — 1 inside dilated GT boxes, 0 outside
    feat: L2-normalized fused features at p2
    """
    feat_norm = F.normalize(feat_p2, dim=1)
    pos = feat_norm[gt_mask.expand_as(feat_norm) > 0.5]
    neg = feat_norm[gt_mask.expand_as(feat_norm) < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.tensor(0.0, device=feat_p2.device)
    return lambda_c * (-pos.mean() + neg.mean())
```

### Add: Gate Regularizer (Phase 3 only)

Prevents the EnergyGate from collapsing to always-thermal or always-RGB:

```python
def gate_regularizer(w_rgb, w_thm, lambda_g=0.05):
    """
    Entropy regularizer: encourage the gate to use BOTH branches
    when both are available (not collapse to 0/1 weights).
    """
    # w_rgb, w_thm: (B, 1, H, W), sum to 1 after softmax
    entropy = -(w_rgb * w_rgb.log().clamp(min=-10) +
                w_thm * w_thm.log().clamp(min=-10))
    return -lambda_g * entropy.mean()   # negative: maximize entropy
```

### Full Loss

```python
L_total = (BBOX_WEIGHT  * L_bbox
         + OBJ_WEIGHT   * L_obj
         + CLASS_WEIGHT * L_cls
         + 0.1          * L_contrast      # Phase 2 onwards
         - 0.05         * L_gate_reg)     # Phase 3 only (negative = maximize entropy)
```

---

## 9. Config Changes Checklist

These are the specific values to change in `config.py` and across the codebase:

```python
# config.py
NUM_ANCHORS = 3          # was 2

# All occurrences in training.py, preprocessing.py, inference.py:
.clamp(-3, 3)            # change to .clamp(-4.5, 4.5)

# inference.py
NMS_IOU_THRESHOLD = 0.35  # was ~0.5 — fixes adjacent person miss
CONF_THRESHOLD = 0.40     # keep or tune slightly lower for jungle

# New architecture modules to add to model.py:
# - EnergyGate  (replaces torch.cat at stem output)
# - BiFusionNeck  (replaces LightweightFPN)
# - ReliabilityClassifier  (replaces IntrusionClassifier, accepts gate weights)
# - AuxSegHead  (training only, removed at export)

# New loss terms to add to training.py:
# - contrast_loss()
# - gate_regularizer()
# - cmm_loss() wrapper
```

---

## 10. Implementation Priorities

Do these in strict order. Each step can be tested independently.

### Step 1 — Immediate (no retraining needed)

Change in `inference.py` right now:
```python
NMS_IOU_THRESHOLD = 0.35    # fixes adjacent person miss immediately
```

Change in `training.py`, `preprocessing.py`, `inference.py`:
```python
.clamp(-4.5, 4.5)            # fixes tall-person head crop immediately
```

Run your existing model on the same diagnostic frames. You will recover some of the FN-1 (adjacent person) and MIS-2 (tall person) errors without any retraining.

### Step 2 — Add CMM to Existing Training (before architecture rebuild)

In `training.py`, wrap the training batch forward pass with the CMM sampler. This costs zero architecture changes and directly addresses the hot-bonnet inconsistency. Even if you run Phase 1 again on LLVIP with CMM enabled, it will significantly reduce the modality-shortcut bias.

### Step 3 — Add EnergyGate and BiFusion Neck

Implement `EnergyGate` and `BiFusionNeck` in `model.py`. Test that the model forward pass runs correctly with both synchronized pairs and single-modality (zeroed) input. Verify shapes match at each scale.

### Step 4 — Re-run K-Means with 3 Anchors

```python
# In preprocessing.py: change num_anchors=3 in analyze_dataset_anchors()
# Re-run: python main.py train --no-kmeans=False
# Check: third anchor should have h/w ratio ~1.2 for side-by-side persons
```

### Step 5 — Phase 1 Retraining (LLVIP, full architecture)

Train the full MicroGhost-V2 from scratch on LLVIP. This takes the same time as your current V1 training. Save checkpoint.

### Step 6 — Phase 2 (WiSARD + ForestPersons + CMM)

Activate all CMM modes, activate contrast loss. Train on the full domain-transfer dataset mix.

### Step 7 — Phase 3 (Camouflage)

Freeze early layers, fine-tune on MCOD + Camo-M3FD + gate regularizer.

### Step 8 — Phase 4 (Polish) → Export → Deploy

---

## Appendix: Forward Pass Pseudocode for V2

```python
def forward(self, x):
    # x: (B, 4, 160, 128) — channels 0:3 = RGB, channel 3: = Thermal
    x_rgb = x[:, :3]       # (B, 3, 160, 128)
    x_thm = x[:, 3:]       # (B, 1, 160, 128)

    # === RGB Branch (fully independent) ===
    feat_rgb = self.rgb_stem(x_rgb)         # (B, 16, 80, 64)
    feat_rgb = self.rgb_scale1(feat_rgb)    # (B, 16, 40, 32)
    feat_rgb_s2 = self.rgb_scale2(feat_rgb) # (B, 12, 20, 16)
    feat_rgb_s3 = self.rgb_scale3(feat_rgb_s2) # (B, 8, 10, 8)

    # === Thermal Branch (fully independent) ===
    feat_thm = self.thm_stem(x_thm)         # (B, 16, 80, 64)
    feat_thm = self.thm_scale1(feat_thm)    # (B, 16, 40, 32)
    feat_thm_s2 = self.thm_scale2(feat_thm) # (B, 12, 20, 16)
    feat_thm_s3 = self.thm_scale3(feat_thm_s2) # (B, 8, 10, 8)

    # === Energy Gate (learned modality weighting at S2) ===
    fused_s2, w_rgb, w_thm = self.energy_gate(feat_rgb_s2, feat_thm_s2)

    # === BiFusion Neck (replaces FPN) ===
    p2, p3 = self.bifusion_neck(fused_s2, feat_rgb_s3, feat_thm_s3)

    # === Detection Heads (3 anchors each) ===
    bbox_small, obj_small = self.head_small(p2)
    bbox_large, obj_large = self.head_large(p3)

    # === Reliability Classifier ===
    label = self.classifier(p2, p3, obj_small, obj_large, w_rgb, w_thm)

    return {
        'bbox_small': bbox_small,
        'obj_small':  obj_small,
        'bbox_large': bbox_large,
        'obj_large':  obj_large,
        'label':      label,
        'w_rgb':      w_rgb,   # used in training loss, excluded at export
        'w_thm':      w_thm,
    }
```

---

*MicroGhost-V2 | Designed for ESP32-S3 PSRAM ≤8MB | Target: INT8 ≤25KB | ≥10 FPS*  
*Architecture status: Design complete — awaiting implementation*
