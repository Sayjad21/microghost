"""
MicroGhost-Thermal: Model Module (V2)
========================================
Dual-branch architecture with late gated fusion for multimodal intrusion detection.

V2 Architecture:
- Dual independent GhostNet+MobileNetV2 branches (RGB + Thermal)
- EnergyGate: learned per-location modality weighting at Scale 2
- BiFusion Neck: bidirectional weighted feature pyramid (replaces FPN)
- ReliabilityClassifier: gate-aware classification (Visible vs Camouflaged)
- AuxSegHead: training-only contrast loss head (zero deployment cost)
- 3 anchors/cell for adjacent person detection

V1 architecture preserved for backward compatibility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    INPUT_SIZE, INPUT_CHANNELS, NUM_CLASSES, NUM_ANCHORS,
    # V1 constants (backward compat)
    STEM_CHANNELS, RGB_STEM_CHANNELS, THERMAL_STEM_CHANNELS,
    SCALE1_CHANNELS, SCALE2_CHANNELS, SCALE3_CHANNELS,
    FPN_CHANNELS, CLASSIFIER_HIDDEN_DIM, EXPAND_RATIO,
    # V2 constants
    V2_STEM_CHANNELS, V2_SCALE1_CHANNELS, V2_SCALE2_CHANNELS,
    V2_SCALE3_CHANNELS, V2_BIFUSION_CHANNELS, V2_CLASSIFIER_HIDDEN_DIM,
    V2_EXPAND_RATIO,
    ESP32_S3,
)


# ============================================================================
# 1. CORE BUILDING BLOCKS (shared V1 + V2)
# ============================================================================

class GhostModule(nn.Module):
    """
    Ghost Module: Generates feature maps using cheap linear operations.
    Reduces computation by ~2x while maintaining representational capacity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_kernel=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                                      kernel_size // 2, bias=False)
        self.primary_bn = nn.BatchNorm2d(init_channels)
        
        # RepGhost parallel 1x1 for richer gradients during training
        self.rep_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(init_channels)
        )
        
        self.primary_act = nn.ReLU6(inplace=True) if relu else nn.Identity()

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel, 1,
                      dw_kernel // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU6(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_bn(self.primary_conv(x))
        if self.training and hasattr(self, 'rep_1x1'):
            x1 = x1 + self.rep_1x1(x)
        x1 = self.primary_act(x1)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: Efficient bottleneck using Ghost modules.
    Structure: Ghost (expansion) → DW Conv → Ghost (projection, linear) → Residual
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 dw_kernel=3, stride=1):
        super().__init__()
        self.stride = stride

        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        if stride > 1:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel, stride,
                          dw_kernel // 2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.conv_dw = nn.Identity()

        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel, stride,
                          dw_kernel // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.conv_dw(x)
        x = self.ghost2(x)
        return x + residual


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block with Linear Bottleneck.
    Structure: Expand → Depthwise → Project (Linear, no ReLU at end)
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 expand_ratio=None):
        super().__init__()
        expand_ratio = expand_ratio or EXPAND_RATIO
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ============================================================================
# 2. V1 ARCHITECTURE (kept for backward compatibility)
# ============================================================================

class LightweightFPN(nn.Module):
    """V1 Lightweight Feature Pyramid Network for multi-scale fusion."""

    def __init__(self, in_channels_s2, in_channels_s3, out_channels=None):
        super().__init__()
        out_channels = out_channels or FPN_CHANNELS

        self.lateral_s2 = nn.Conv2d(in_channels_s2, out_channels, 1,
                                    bias=False)
        self.lateral_s3 = nn.Conv2d(in_channels_s3, out_channels, 1,
                                    bias=False)

        self.smooth_s2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.smooth_s3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, feat_s2, feat_s3):
        lat_s2 = self.lateral_s2(feat_s2)
        lat_s3 = self.lateral_s3(feat_s3)
        upsampled_s3 = F.interpolate(lat_s3, size=lat_s2.shape[2:],
                                     mode='nearest')
        p2 = self.smooth_s2(lat_s2 + upsampled_s3)
        p3 = self.smooth_s3(lat_s3)
        return p2, p3


class SSDLiteHead(nn.Module):
    """SSDLite Detection Head using depthwise-separable convolutions."""

    def __init__(self, in_channels, num_anchors=None):
        super().__init__()
        num_anchors = num_anchors or NUM_ANCHORS

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )

        self.bbox_head = nn.Conv2d(in_channels, num_anchors * 4, 1)
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)

    def forward(self, x):
        feat = self.feature(x)
        bbox = self.bbox_head(feat)
        obj = self.obj_head(feat)
        return bbox, obj


class IntrusionClassifier(nn.Module):
    """V1 Multiclass intrusion classifier with objectness-weighted attention."""

    def __init__(self, in_channels, num_classes=None, hidden_dim=None):
        super().__init__()
        num_classes = num_classes or NUM_CLASSES
        hidden_dim = hidden_dim or CLASSIFIER_HIDDEN_DIM

        self.modality_attention = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_dim // 2, in_channels * 2),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat_p2, feat_p3, obj_p2, obj_p3):
        attn_p2 = torch.sigmoid(
            obj_p2.max(dim=1, keepdim=True)[0]
        )
        attn_p3 = torch.sigmoid(
            obj_p3.max(dim=1, keepdim=True)[0]
        )

        p2_w = (feat_p2 * attn_p2).sum(dim=[2, 3]) / \
               (attn_p2.sum(dim=[2, 3]) + 1e-6)
        p3_w = (feat_p3 * attn_p3).sum(dim=[2, 3]) / \
               (attn_p3.sum(dim=[2, 3]) + 1e-6)

        combined = torch.cat([p2_w, p3_w], dim=1)

        attn_weights = self.modality_attention(combined)
        attended_features = combined * attn_weights

        return self.classifier(attended_features)


class MicroGhostThermal(nn.Module):
    """V1 MicroGhost-Thermal: Early fusion, shared backbone, FPN."""

    def __init__(self, num_classes=None, num_anchors=None,
                 input_size=None, classifier_hidden_dim=None):
        super().__init__()
        num_classes = num_classes or NUM_CLASSES
        num_anchors = num_anchors or NUM_ANCHORS
        input_size = input_size or INPUT_SIZE
        classifier_hidden_dim = classifier_hidden_dim or CLASSIFIER_HIDDEN_DIM

        self.input_size = input_size
        if isinstance(input_size, tuple):
            self.input_h, self.input_w = input_size
        else:
            self.input_h, self.input_w = input_size, input_size
        self.num_classes = num_classes
        self.classifier_hidden_dim = classifier_hidden_dim

        # Dual Stem
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            GhostModule(8, RGB_STEM_CHANNELS, kernel_size=1, stride=1),
        )

        self.thermal_stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            GhostModule(8, THERMAL_STEM_CHANNELS, kernel_size=1, stride=1),
        )

        # Shared backbone
        self.scale1 = nn.Sequential(
            GhostBottleneck(STEM_CHANNELS, STEM_CHANNELS * 2,
                            SCALE1_CHANNELS, stride=2),
            GhostBottleneck(SCALE1_CHANNELS, SCALE1_CHANNELS * 2,
                            SCALE1_CHANNELS, stride=1),
        )

        self.scale2 = nn.Sequential(
            InvertedResidual(SCALE1_CHANNELS, SCALE2_CHANNELS,
                             stride=2, expand_ratio=EXPAND_RATIO),
            InvertedResidual(SCALE2_CHANNELS, SCALE2_CHANNELS,
                             stride=1, expand_ratio=EXPAND_RATIO),
        )

        self.scale3 = nn.Sequential(
            InvertedResidual(SCALE2_CHANNELS, SCALE3_CHANNELS,
                             stride=2, expand_ratio=EXPAND_RATIO),
            InvertedResidual(SCALE3_CHANNELS, SCALE3_CHANNELS,
                             stride=1, expand_ratio=EXPAND_RATIO),
        )

        self.fpn = LightweightFPN(
            in_channels_s2=SCALE2_CHANNELS,
            in_channels_s3=SCALE3_CHANNELS,
            out_channels=FPN_CHANNELS,
        )

        self.head_small = SSDLiteHead(FPN_CHANNELS, num_anchors=num_anchors)
        self.head_large = SSDLiteHead(FPN_CHANNELS, num_anchors=num_anchors)

        self.classifier = IntrusionClassifier(
            in_channels=FPN_CHANNELS,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x_rgb = x[:, :3, :, :]
        x_thermal = x[:, 3:, :, :]

        feat_rgb = self.rgb_stem(x_rgb)
        feat_thermal = self.thermal_stem(x_thermal)
        feat_fused = torch.cat([feat_rgb, feat_thermal], dim=1)

        s1 = self.scale1(feat_fused)
        s2 = self.scale2(s1)
        s3 = self.scale3(s2)

        p2, p3 = self.fpn(s2, s3)

        bbox_small, obj_small = self.head_small(p2)
        bbox_large, obj_large = self.head_large(p3)

        label = self.classifier(p2, p3, obj_small, obj_large)

        return {
            'bbox_small': bbox_small,
            'obj_small': obj_small,
            'bbox_large': bbox_large,
            'obj_large': obj_large,
            'label': label,
        }


# ============================================================================
# 3. V2 NEW MODULES
# ============================================================================

class EnergyGate(nn.Module):
    """
    Per-location gating between RGB and Thermal at Scale 2.

    Computes learned energy projections and applies softmax across the two
    branches at each spatial location. This lets the network suppress
    whichever branch has unreliable content (e.g., dark RGB at night,
    hot car bonnet in thermal).

    ~30 parameters. Applied BEFORE BiFusion Neck.
    """

    def __init__(self, channels):
        super().__init__()
        self.proj_rgb = nn.Conv2d(channels, 1, 1, bias=True)
        self.proj_thm = nn.Conv2d(channels, 1, 1, bias=True)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, feat_rgb, feat_thm):
        e_rgb = self.proj_rgb(feat_rgb)          # (B, 1, H, W)
        e_thm = self.proj_thm(feat_thm)          # (B, 1, H, W)
        
        temp = torch.clamp(self.temperature, min=0.5, max=5.0)
        
        weights = torch.softmax(
            torch.stack([e_rgb, e_thm], dim=1) / temp,  # (B, 2, 1, H, W)
            dim=1
        )
        w_rgb = weights[:, 0]                    # (B, 1, H, W)
        w_thm = weights[:, 1]                    # (B, 1, H, W)
        fused = w_rgb * feat_rgb + w_thm * feat_thm
        return fused, w_rgb, w_thm


class BiFusionNeck(nn.Module):
    """
    Bidirectional weighted feature pyramid (replaces LightweightFPN).

    Receives S2 fused features (from EnergyGate) and S3 features from
    both branches separately. Uses learned normalized weights (BiFPN-style)
    for top-down and bottom-up passes.

    Inputs:
        fused_s2:    (B, s2_ch, 16, 20) — gated EnergyGate output
        feat_rgb_s3: (B, s3_ch, 8, 10)
        feat_thm_s3: (B, s3_ch, 8, 10)

    Outputs:
        p2: (B, out_ch, 16, 20) — small/distant target features
        p3: (B, out_ch, 8, 10)  — large/close target features
    """

    def __init__(self, s2_ch=None, s3_ch=None, out_ch=None):
        super().__init__()
        s2_ch = s2_ch or V2_SCALE2_CHANNELS
        s3_ch = s3_ch or V2_SCALE3_CHANNELS
        out_ch = out_ch or V2_BIFUSION_CHANNELS

        # Lateral projections to unified channel count
        self.lat_s2 = nn.Conv2d(s2_ch, out_ch, 1, bias=False)
        self.lat_rgb = nn.Conv2d(s3_ch, out_ch, 1, bias=False)
        self.lat_thm = nn.Conv2d(s3_ch, out_ch, 1, bias=False)

        # Learned BiFPN weights (softmax-normalized)
        # Top-down: P3 = w1*rgb_s3 + w2*thm_s3
        self.w_td = nn.Parameter(torch.zeros(2))

        # Bottom-up P2: w3*s2 + w4*P3_upsampled
        self.w_bu = nn.Parameter(torch.zeros(2))

        # DW-separable refinement convolutions
        self.refine_p3 = self._dw_sep(out_ch, out_ch)
        self.refine_p2 = self._dw_sep(out_ch, out_ch)

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
        lat_s2 = self.lat_s2(fused_s2)          # (B, CH, 16, 20)
        lat_rgb = self.lat_rgb(feat_rgb_s3)      # (B, CH, 8, 10)
        lat_thm = self.lat_thm(feat_thm_s3)      # (B, CH, 8, 10)

        # ── Top-down: fuse S3 from both branches ──
        w_td = F.softplus(self.w_td) + eps
        w_td = w_td / w_td.sum()
        p3_td = self.refine_p3(
            w_td[0] * lat_rgb + w_td[1] * lat_thm
        )                                        # (B, CH, 8, 10)

        # ── Bottom-up: upsample P3 and merge with S2 ──
        p3_up = F.interpolate(p3_td, size=lat_s2.shape[2:], mode='nearest')
        w_bu = F.softplus(self.w_bu) + eps
        w_bu = w_bu / w_bu.sum()
        p2_out = self.refine_p2(
            w_bu[0] * lat_s2 + w_bu[1] * p3_up
        )                                        # (B, CH, 16, 20)

        return p2_out, p3_td


class ReliabilityClassifier(nn.Module):
    """
    Gate-aware classifier for V2 (replaces IntrusionClassifier).

    Receives the EnergyGate weights as auxiliary input, enabling
    Visible vs Camouflaged classification based on which modality
    was dominant at detection time.

    - w_thm >> w_rgb → thermal dominant → likely Person_Camouflaged
    - w_rgb ≈ w_thm → both agree → Person_Visible
    """

    def __init__(self, in_channels=None, num_classes=None, hidden_dim=None):
        super().__init__()
        in_channels = in_channels or V2_BIFUSION_CHANNELS
        num_classes = num_classes or NUM_CLASSES
        hidden_dim = hidden_dim or V2_CLASSIFIER_HIDDEN_DIM

        # Modality-gate-aware attention (+2 for gate weight scalars)
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
            nn.Linear(hidden_dim, num_classes + 1), # +1 for IoU regression
        )

    def forward(self, feat_p2, feat_p3, obj_p2, obj_p3, w_rgb, w_thm):
        # Objectness-weighted spatial pooling
        attn_p2 = torch.sigmoid(obj_p2.max(dim=1, keepdim=True)[0])
        attn_p3 = torch.sigmoid(obj_p3.max(dim=1, keepdim=True)[0])
        p2_w = (feat_p2 * attn_p2).sum([2, 3]) / (attn_p2.sum([2, 3]) + 1e-6)
        p3_w = (feat_p3 * attn_p3).sum([2, 3]) / (attn_p3.sum([2, 3]) + 1e-6)
        combined = torch.cat([p2_w, p3_w], dim=1)   # (B, 2C)

        # Gate summary: mean weight across spatial dimensions → (B, 2)
        gate_rgb_mean = w_rgb.mean(dim=[1, 2, 3])   # (B,)
        gate_thm_mean = w_thm.mean(dim=[1, 2, 3])   # (B,)
        gate_summary = torch.stack([gate_rgb_mean, gate_thm_mean], dim=1)  # (B, 2)

        # Attend features using gate-aware attention
        gate_input = torch.cat([combined, gate_summary], dim=1)  # (B, 2C+2)
        attn_weights = self.modality_gate(gate_input)  # (B, 2C)
        attended = combined * attn_weights

        return self.classifier(attended)


class AuxSegHead(nn.Module):
    """
    Training-only auxiliary segmentation head for TFDet-style contrast loss.
    Removed at export. Zero deployment cost.

    Produces per-spatial-location person/background logits from p2 features.
    """

    def __init__(self, in_ch=None):
        super().__init__()
        in_ch = in_ch or V2_BIFUSION_CHANNELS
        self.proj = nn.Conv2d(in_ch, 1, 1)

    def forward(self, feat_p2):
        return self.proj(feat_p2)   # (B, 1, H, W)


# ============================================================================
# 4. V2 COMPLETE MODEL: MicroGhostV2
# ============================================================================

class MicroGhostV2(nn.Module):
    """
    MicroGhost-V2: Asynchronous Dual-Branch Architecture.

    Key changes vs V1:
    - Parallel independent RGB + Thermal branches (no shared weights)
    - EnergyGate at Scale 2 for learned modality weighting
    - BiFusion Neck (bidirectional weighted pyramid, replaces FPN)
    - ReliabilityClassifier with gate-weight awareness
    - AuxSegHead for training-only contrast loss
    - 3 anchors per cell
    - Graceful camera failure (either branch can operate independently)

    Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │  RGB (3ch)            Thermal (1ch)                        │
    │     │                      │                               │
    │  RGB Stem(16)          Thm Stem(16)                        │
    │     │                      │                               │
    │  RGB Scale1(24)        Thm Scale1(24)                      │
    │     │                      │                               │
    │  RGB Scale2(32) ──► EnergyGate ◄── Thm Scale2(32)          │
    │                       │ fused(32)                           │
    │  RGB Scale3(48) ──► BiFusion Neck ◄── Thm Scale3(48)       │
    │                    ┌────┴────┐                             │
    │                 p2(48)    p3(48)                            │
    │              SmallHead  LargeHead                          │
    │                    └────┬────┘                             │
    │              ReliabilityClassifier                         │
    └────────────────────────────────────────────────────────────┘
    """

    def __init__(self, num_classes=None, num_anchors=None,
                 input_size=None, classifier_hidden_dim=None,
                 training_mode=True):
        super().__init__()
        num_classes = num_classes or NUM_CLASSES
        num_anchors = num_anchors or NUM_ANCHORS
        input_size = input_size or INPUT_SIZE
        classifier_hidden_dim = classifier_hidden_dim or V2_CLASSIFIER_HIDDEN_DIM

        self.input_size = input_size
        if isinstance(input_size, tuple):
            self.input_h, self.input_w = input_size
        else:
            self.input_h, self.input_w = input_size, input_size
        self.num_classes = num_classes
        self.classifier_hidden_dim = classifier_hidden_dim
        self.training_mode = training_mode

        S = V2_STEM_CHANNELS
        S1 = V2_SCALE1_CHANNELS
        S2 = V2_SCALE2_CHANNELS
        S3 = V2_SCALE3_CHANNELS
        E = V2_EXPAND_RATIO

        # ========== RGB BRANCH (fully independent) ==========
        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            GhostModule(8, S, kernel_size=1, stride=1),
        )
        self.rgb_scale1 = nn.Sequential(
            GhostBottleneck(S, S * 2, S1, stride=2),
            GhostBottleneck(S1, S1 * 2, S1, stride=1),
        )
        self.rgb_scale2 = nn.Sequential(
            InvertedResidual(S1, S2, stride=2, expand_ratio=E),
            InvertedResidual(S2, S2, stride=1, expand_ratio=E),
        )
        self.rgb_scale3 = nn.Sequential(
            InvertedResidual(S2, S3, stride=2, expand_ratio=E),
            InvertedResidual(S3, S3, stride=1, expand_ratio=E),
        )

        # ========== THERMAL BRANCH (fully independent) ==========
        self.thm_stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            GhostModule(8, S, kernel_size=1, stride=1),
        )
        self.thm_scale1 = nn.Sequential(
            GhostBottleneck(S, S * 2, S1, stride=2),
            GhostBottleneck(S1, S1 * 2, S1, stride=1),
        )
        self.thm_scale2 = nn.Sequential(
            InvertedResidual(S1, S2, stride=2, expand_ratio=E),
            InvertedResidual(S2, S2, stride=1, expand_ratio=E),
        )
        self.thm_scale3 = nn.Sequential(
            InvertedResidual(S2, S3, stride=2, expand_ratio=E),
            InvertedResidual(S3, S3, stride=1, expand_ratio=E),
        )

        # ========== ENERGY GATE (at Scale 2 output) ==========
        self.energy_gate = EnergyGate(channels=S2)

        # ========== BIFUSION NECK (replaces FPN) ==========
        self.bifusion_neck = BiFusionNeck(
            s2_ch=S2, s3_ch=S3, out_ch=V2_BIFUSION_CHANNELS,
        )

        # ========== DETECTION HEADS (3 anchors each) ==========
        self.head_small = SSDLiteHead(V2_BIFUSION_CHANNELS, num_anchors=num_anchors)
        self.head_large = SSDLiteHead(V2_BIFUSION_CHANNELS, num_anchors=num_anchors)

        # ========== RELIABILITY CLASSIFIER ==========
        self.classifier = ReliabilityClassifier(
            in_channels=V2_BIFUSION_CHANNELS,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
        )

        # ========== AUX SEGMENTATION HEAD (training only) ==========
        self.aux_seg_head = AuxSegHead(in_ch=V2_BIFUSION_CHANNELS)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass for V2 dual-branch architecture.

        Args:
            x: (B, 4, H, W) float tensor (channels 0:3 = RGB, channel 3: = Thermal)

        Returns:
            dict with detection outputs + gate weights + aux seg logits
        """
        x_rgb = x[:, :3]       # (B, 3, H, W)
        x_thm = x[:, 3:]       # (B, 1, H, W)

        # Modality masking to prevent BatchNorm shift artifacts on empty inputs
        rgb_present = (x_rgb.abs().mean(dim=[1,2,3], keepdim=True) > 1e-5).float()
        thm_present = (x_thm.abs().mean(dim=[1,2,3], keepdim=True) > 1e-5).float()

        # === RGB Branch (fully independent) ===
        feat_rgb = self.rgb_stem(x_rgb)             # (B, 16, 64, 80)
        feat_rgb = self.rgb_scale1(feat_rgb)        # (B, 24, 32, 40)
        feat_rgb_s2 = self.rgb_scale2(feat_rgb)     # (B, 32, 16, 20)
        feat_rgb_s3 = self.rgb_scale3(feat_rgb_s2)  # (B, 48, 8, 10)
        
        feat_rgb_s2 = feat_rgb_s2 * rgb_present
        feat_rgb_s3 = feat_rgb_s3 * rgb_present

        # === Thermal Branch (fully independent) ===
        feat_thm = self.thm_stem(x_thm)             # (B, 16, 64, 80)
        feat_thm = self.thm_scale1(feat_thm)        # (B, 24, 32, 40)
        feat_thm_s2 = self.thm_scale2(feat_thm)     # (B, 32, 16, 20)
        feat_thm_s3 = self.thm_scale3(feat_thm_s2)  # (B, 48, 8, 10)
        
        feat_thm_s2 = feat_thm_s2 * thm_present
        feat_thm_s3 = feat_thm_s3 * thm_present

        # === Energy Gate (learned modality weighting at S2) ===
        fused_s2, w_rgb, w_thm = self.energy_gate(feat_rgb_s2, feat_thm_s2)

        # === BiFusion Neck (replaces FPN) ===
        p2, p3 = self.bifusion_neck(fused_s2, feat_rgb_s3, feat_thm_s3)

        # === Detection Heads (3 anchors each) ===
        bbox_small, obj_small = self.head_small(p2)
        bbox_large, obj_large = self.head_large(p3)

        # === Reliability Classifier ===
        label = self.classifier(p2, p3, obj_small, obj_large, w_rgb, w_thm)

        result = {
            'bbox_small': bbox_small,
            'obj_small':  obj_small,
            'bbox_large': bbox_large,
            'obj_large':  obj_large,
            'label':      label,
            'w_rgb':      w_rgb,
            'w_thm':      w_thm,
        }

        # Aux seg head (training only — removed at export)
        if self.training_mode and self.training:
            result['aux_seg'] = self.aux_seg_head(p2)

        return result

    def freeze_early_layers(self):
        """Freeze stems and Scale 1 for Phase 3 fine-tuning."""
        for name, param in self.named_parameters():
            if any(prefix in name for prefix in
                   ['rgb_stem', 'thm_stem', 'rgb_scale1', 'thm_scale1']):
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for Phase 4 polish."""
        for param in self.parameters():
            param.requires_grad = True


# ============================================================================
# 5. MODEL ANALYSIS UTILITIES
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    """Count all parameters (including frozen)."""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size(model):
    """Estimate model size in different quantization formats."""
    param_count = count_all_parameters(model)
    fp32_mb = param_count * 4 / (1024 * 1024)
    int8_kb = param_count * 1 / 1024
    return param_count, fp32_mb, int8_kb


def estimate_peak_sram(model, input_size=None, batch_size=1):
    """Estimate peak SRAM usage during inference on ESP32-S3."""
    input_size = input_size or INPUT_SIZE
    activations = []

    def hook_fn(module, inp, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.numel())

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU6)):
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        h, w = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        x = torch.randn(batch_size, INPUT_CHANNELS, h, w)
        _ = model(x)

    for hook in hooks:
        hook.remove()

    input_bytes = batch_size * INPUT_CHANNELS * h * w
    max_activation_int8 = max(activations) if activations else 0

    return {
        'input_buffer_kb': input_bytes / 1024,
        'peak_activation_fp32_kb': max(activations) * 4 / 1024 if activations else 0,
        'peak_activation_int8_kb': max_activation_int8 / 1024,
        'total_arena_int8_kb': (input_bytes + max_activation_int8) / 1024,
        'fits_esp32_s3': (input_bytes + max_activation_int8) < ESP32_S3['max_arena_sram_kb'] * 1024,
    }


def print_model_analysis(model):
    """Print comprehensive model analysis for V1 or V2."""
    param_count, fp32_mb, int8_kb = estimate_model_size(model)
    sram = estimate_peak_sram(model)

    is_v2 = isinstance(model, MicroGhostV2)
    model_name = "MicroGhost-V2" if is_v2 else "MicroGhost-V1"

    print(f"\n[OK] {model_name}:")
    print(f"   Parameters:       {param_count:,}")
    print(f"   Trainable:        {count_parameters(model):,}")
    print(f"   Est. Size (FP32): {param_count * 4 / 1024:,.1f} KB")
    print(f"   Est. Size (FP16): {param_count * 2 / 1024:,.1f} KB")
    print(f"   Est. Size (INT8): {int8_kb:,.1f} KB")
    print()
    print(f"  Input buffer:      {sram['input_buffer_kb']:>10.1f} KB")
    print(f"  Peak act (FP32):   {sram['peak_activation_fp32_kb']:>10.1f} KB")
    print(f"  Peak act (INT8):   {sram['peak_activation_int8_kb']:>10.1f} KB")
    print(f"  Total arena INT8:  {sram['total_arena_int8_kb']:>10.1f} KB")
    print(f"  Fits ESP32-S3:     {'OK' if sram['fits_esp32_s3'] else 'FAIL'} "
          f"(limit: {ESP32_S3['max_arena_sram_kb']}KB)")

    # Layer-by-layer breakdown
    print(f"\n  {'Layer':<35} {'Params':>12} {'Size (KB)':>10}")
    print("  " + "-" * 59)

    if is_v2:
        components = [
            ('RGB Stem', model.rgb_stem),
            ('RGB Scale 1 (Ghost)', model.rgb_scale1),
            ('RGB Scale 2 (InvRes)', model.rgb_scale2),
            ('RGB Scale 3 (InvRes)', model.rgb_scale3),
            ('Thm Stem', model.thm_stem),
            ('Thm Scale 1 (Ghost)', model.thm_scale1),
            ('Thm Scale 2 (InvRes)', model.thm_scale2),
            ('Thm Scale 3 (InvRes)', model.thm_scale3),
            ('Energy Gate', model.energy_gate),
            ('BiFusion Neck', model.bifusion_neck),
            ('Head Small', model.head_small),
            ('Head Large', model.head_large),
            ('Classifier', model.classifier),
            ('Aux Seg Head (train only)', model.aux_seg_head),
        ]
    else:
        components = [
            ('RGB Stem', model.rgb_stem),
            ('Thermal Stem', model.thermal_stem),
            ('Scale 1 (Ghost)', model.scale1),
            ('Scale 2 (InvRes)', model.scale2),
            ('Scale 3 (InvRes)', model.scale3),
            ('FPN', model.fpn),
            ('Head Small', model.head_small),
            ('Head Large', model.head_large),
            ('Classifier', model.classifier),
        ]

    total = 0
    for name, module in components:
        params = sum(p.numel() for p in module.parameters())
        kb = params * 4 / 1024
        total += params
        print(f"  {name:<35} {params:>12,} {kb:>8.1f} KB")

    print("  " + "-" * 59)
    print(f"  {'TOTAL':<35} {total:>12,} {total * 4 / 1024:>8.1f} KB")


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Model Module — V2 Self Test")
    print("=" * 60)

    # Test V2 model
    model = MicroGhostV2()
    print_model_analysis(model)

    # Test forward pass (training mode)
    model.train()
    h, w = INPUT_SIZE if isinstance(INPUT_SIZE, tuple) else (INPUT_SIZE, INPUT_SIZE)
    dummy = torch.randn(2, INPUT_CHANNELS, h, w)
    with torch.no_grad():
        outputs = model(dummy)

    print(f"\n[OK] V2 Forward Pass (training):")
    for key, val in outputs.items():
        print(f"    {key}: {val.shape}")

    # Test forward pass (eval mode — no aux_seg)
    model.eval()
    with torch.no_grad():
        outputs_eval = model(dummy)
    print(f"\n[OK] V2 Forward Pass (eval):")
    for key, val in outputs_eval.items():
        print(f"    {key}: {val.shape}")
    assert 'aux_seg' not in outputs_eval, "aux_seg should not be in eval outputs"

    # Test single-modality (CMM-RXTO: thermal zeroed)
    dummy_rxto = dummy.clone()
    dummy_rxto[:, 3:] = 0.0
    model.eval()
    with torch.no_grad():
        outputs_rxto = model(dummy_rxto)
    print(f"\n[OK] CMM-RXTO (thermal zeroed) — forward pass OK")

    # Test single-modality (CMM-ROTX: RGB zeroed)
    dummy_rotx = dummy.clone()
    dummy_rotx[:, :3] = 0.0
    with torch.no_grad():
        outputs_rotx = model(dummy_rotx)
    print(f"[OK] CMM-ROTX (RGB zeroed) — forward pass OK")

    # Test freeze/unfreeze
    model.freeze_early_layers()
    trainable_after_freeze = count_parameters(model)
    model.unfreeze_all()
    trainable_after_unfreeze = count_parameters(model)
    print(f"\n[OK] Freeze test: {trainable_after_freeze:,} trainable (frozen) -> "
          f"{trainable_after_unfreeze:,} trainable (unfrozen)")

    # Verify gate weights sum to ~1
    w_rgb_mean = outputs_eval['w_rgb'].mean().item()
    w_thm_mean = outputs_eval['w_thm'].mean().item()
    print(f"\n[OK] Gate weights: w_rgb={w_rgb_mean:.4f}, w_thm={w_thm_mean:.4f}, "
          f"sum={w_rgb_mean + w_thm_mean:.4f}")

    print("\n[OK] All V2 model tests passed!")
