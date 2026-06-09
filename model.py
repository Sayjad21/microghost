"""
MicroGhost-Thermal: Model Module
==================================
Lightweight neural network for multimodal intrusion detection on ESP32-S3.

Architecture: Dual GhostNet Stems + MobileNetV2 blocks + FPN + dual SSDLite heads
Target: 8MB PSRAM, ~2500KB FP16, ≥10 FPS on ESP32-S3

Key differences from MicroGhost-Hand:
- Dual-channel input (RGB + Thermal)
- Perfectly sized channels to meet ~67k parameter budget
- 2 anchors/cell instead of 3
- 4-class classification (background, visible, camouflaged, vehicle)
- 160x128 input
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    INPUT_SIZE, INPUT_CHANNELS, NUM_CLASSES, NUM_ANCHORS,
    STEM_CHANNELS, RGB_STEM_CHANNELS, THERMAL_STEM_CHANNELS, 
    SCALE1_CHANNELS, SCALE2_CHANNELS, SCALE3_CHANNELS,
    FPN_CHANNELS, CLASSIFIER_HIDDEN_DIM, EXPAND_RATIO,
    ESP32_S3,
)


# ============================================================================
# 1. CORE BUILDING BLOCKS
# ============================================================================

class GhostModule(nn.Module):
    """
    Ghost Module: Generates feature maps using cheap linear operations.

    Instead of expensive convolutions for all output channels:
    1. Generate 'intrinsic' features with standard conv (half channels)
    2. Generate 'ghost' features with cheap depthwise conv
    3. Concatenate both for full output

    Reduces computation by ~2x while maintaining representational capacity.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_kernel=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True) if relu else nn.Identity(),
        )

        # Cheap operation (ghost features via depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel, 1,
                      dw_kernel // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU6(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: Efficient bottleneck using Ghost modules.

    Structure:
    - Ghost Module (expansion)
    - Depthwise Conv (spatial mixing, optional stride)
    - Ghost Module (projection, NO ReLU — Linear Bottleneck)
    - Residual connection
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 dw_kernel=3, stride=1):
        super().__init__()
        self.stride = stride

        # Ghost expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # Depthwise convolution (for stride > 1)
        if stride > 1:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel, stride,
                          dw_kernel // 2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.conv_dw = nn.Identity()

        # Ghost projection (Linear Bottleneck — no ReLU!)
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # Shortcut connection
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

    Why Linear Bottleneck matters for MCUs:
    - ReLU destroys information in low-dimensional spaces
    - During INT8 quantization, this information loss compounds
    - Linear bottleneck (no ReLU at end) preserves gradient flow

    Structure: Expand → Depthwise → Project (Linear)
    Uses ReLU6 for bounded activations (INT8 quantization friendly).
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 expand_ratio=None):
        super().__init__()
        expand_ratio = expand_ratio or EXPAND_RATIO
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        layers = []

        # Expansion (only if expand_ratio > 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Projection (LINEAR — no activation!)
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
# 2. LIGHTWEIGHT FEATURE PYRAMID NETWORK (FPN)
# ============================================================================

class LightweightFPN(nn.Module):
    """
    Lightweight Feature Pyramid Network for multi-scale fusion.

    Fuses features from Scale 2 and Scale 3:
    - Upsamples deep features → adds to shallow features
    - Depthwise-separable convolution for efficiency
    - Produces enhanced features at both scales

    Critical for detecting intruders at varying distances from sensor.
    """

    def __init__(self, in_channels_s2, in_channels_s3, out_channels=None):
        super().__init__()
        out_channels = out_channels or FPN_CHANNELS

        # Lateral connections (1×1 conv to unify channels)
        self.lateral_s2 = nn.Conv2d(in_channels_s2, out_channels, 1,
                                    bias=False)
        self.lateral_s3 = nn.Conv2d(in_channels_s3, out_channels, 1,
                                    bias=False)

        # Top-down pathway (depthwise-separable)
        self.smooth_s2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Refinement for Scale 3
        self.smooth_s3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, feat_s2, feat_s3):
        """
        Args:
            feat_s2: (B, C_s2, H2, W2) — higher resolution features
            feat_s3: (B, C_s3, H3, W3) — deeper semantic features
        Returns:
            p2: Enhanced higher-res features (small/distant targets)
            p3: Enhanced lower-res features (large/close targets)
        """
        lat_s2 = self.lateral_s2(feat_s2)
        lat_s3 = self.lateral_s3(feat_s3)

        # Upsample s3 and fuse with s2
        upsampled_s3 = F.interpolate(lat_s3, size=lat_s2.shape[2:],
                                     mode='nearest')
        p2 = self.smooth_s2(lat_s2 + upsampled_s3)
        p3 = self.smooth_s3(lat_s3)

        return p2, p3


# ============================================================================
# 3. DETECTION HEAD (SSDLite Style)
# ============================================================================

class SSDLiteHead(nn.Module):
    """
    SSDLite Detection Head using depthwise-separable convolutions.

    Each anchor predicts:
    - 4 values: (cx_offset, cy_offset, log_w, log_h)
    - 1 value: objectness score (logit)

    Uses ReLU6 for INT8 quantization compatibility.
    """

    def __init__(self, in_channels, num_anchors=None):
        super().__init__()
        num_anchors = num_anchors or NUM_ANCHORS

        # Shared feature extraction
        self.feature = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, 1, 1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )

        # BBox regression: 4 values per anchor
        self.bbox_head = nn.Conv2d(in_channels, num_anchors * 4, 1)

        # Objectness: 1 value per anchor
        self.obj_head = nn.Conv2d(in_channels, num_anchors, 1)

    def forward(self, x):
        """
        Returns:
            bbox: (B, num_anchors * 4, H, W)
            obj:  (B, num_anchors, H, W)
        """
        feat = self.feature(x)
        bbox = self.bbox_head(feat)
        obj = self.obj_head(feat)
        return bbox, obj


# ============================================================================
# 4. INTRUSION CLASSIFIER
# ============================================================================

class IntrusionClassifier(nn.Module):
    """
    Multiclass intrusion classifier with objectness-weighted attention.

    Uses detection head objectness maps as spatial attention weights:
    - Areas where model detects a target contribute more
    - Background regions are suppressed
    - Concatenates features from both pyramid levels
    - Employs an implicit modality attention to differentiate Visible vs Camouflaged

    Output: 4 classes (Background, Person_Visible, Person_Camouflaged, Vehicle_Boat)
    """

    def __init__(self, in_channels, num_classes=None, hidden_dim=None):
        super().__init__()
        num_classes = num_classes or NUM_CLASSES
        hidden_dim = hidden_dim or CLASSIFIER_HIDDEN_DIM

        # Input is 2× in_channels (concatenation of p2 and p3)
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
        """
        Objectness-weighted pooling for classification.

        Args:
            feat_p2: (B, C, H2, W2) — fine detail features
            feat_p3: (B, C, H3, W3) — semantic features
            obj_p2:  (B, A, H2, W2) — objectness logits (small head)
            obj_p3:  (B, A, H3, W3) — objectness logits (large head)

        Returns:
            (B, num_classes) logits
        """
        # Spatial attention from objectness (max across anchors)
        attn_p2 = torch.sigmoid(
            obj_p2.max(dim=1, keepdim=True)[0]
        )  # (B, 1, H2, W2)
        attn_p3 = torch.sigmoid(
            obj_p3.max(dim=1, keepdim=True)[0]
        )  # (B, 1, H3, W3)

        # Weighted average pooling
        p2_w = (feat_p2 * attn_p2).sum(dim=[2, 3]) / \
               (attn_p2.sum(dim=[2, 3]) + 1e-6)
        p3_w = (feat_p3 * attn_p3).sum(dim=[2, 3]) / \
               (attn_p3.sum(dim=[2, 3]) + 1e-6)

        # Concatenate
        combined = torch.cat([p2_w, p3_w], dim=1)  # (B, 2C)
        
        # Cross-modality feature attention (helps separate camouflaged vs visible)
        attn_weights = self.modality_attention(combined)
        attended_features = combined * attn_weights
        
        return self.classifier(attended_features)


# ============================================================================
# 5. COMPLETE MODEL: MicroGhostThermal
# ============================================================================

class MicroGhostThermal(nn.Module):
    """
    MicroGhost-Thermal: Complete model for thermal intrusion detection.

    Designed for ESP32-S3 deployment with:
    - GhostNet efficiency (cheap feature generation)
    - MobileNetV2 linear bottlenecks (quantization-friendly)
    - Lightweight FPN (multi-scale fusion)
    - Dual SSDLite heads (small + large target detection)
    - Binary intrusion classifier with attention pooling
    - ReLU6 everywhere (bounded [0,6] for INT8 quantization)

    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │ Input: 64×64×1 thermal                                  │
    │                                                         │
    │ Stem:    1→16ch, stride 2  → 32×32                      │
    │ Scale1: 16→16ch, stride 2  → 16×16                      │
    │ Scale2: 16→12ch, stride 2  →  8×8  ──┐                  │
    │ Scale3: 12→ 8ch, stride 2  →  4×4  ──┤                  │
    │                                       │                  │
    │ FPN: (12,8) → 12ch                    │                  │
    │   ├─ p2: 8×8  → SSDLite Head (small)  │                  │
    │   └─ p3: 4×4  → SSDLite Head (large)  │                  │
    │                                       │                  │
    │ Classifier: attention-weighted binary  │                  │
    └─────────────────────────────────────────────────────────┘
    """

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

        # ========== DUAL STEM (160x128 → 80x64) ==========
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
        
        # Fusion is just concatenation in forward pass
        # Fused channels = RGB_STEM_CHANNELS + THERMAL_STEM_CHANNELS = STEM_CHANNELS

        # ========== SCALE 1: Ghost Bottleneck (32→16) ==========
        self.scale1 = nn.Sequential(
            GhostBottleneck(STEM_CHANNELS, STEM_CHANNELS * 2,
                            SCALE1_CHANNELS, stride=2),
            GhostBottleneck(SCALE1_CHANNELS, SCALE1_CHANNELS * 2,
                            SCALE1_CHANNELS, stride=1),
        )

        # ========== SCALE 2: Inverted Residual (16→8) ==========
        self.scale2 = nn.Sequential(
            InvertedResidual(SCALE1_CHANNELS, SCALE2_CHANNELS,
                             stride=2, expand_ratio=EXPAND_RATIO),
            InvertedResidual(SCALE2_CHANNELS, SCALE2_CHANNELS,
                             stride=1, expand_ratio=EXPAND_RATIO),
        )

        # ========== SCALE 3: Inverted Residual (8→4) ==========
        self.scale3 = nn.Sequential(
            InvertedResidual(SCALE2_CHANNELS, SCALE3_CHANNELS,
                             stride=2, expand_ratio=EXPAND_RATIO),
            InvertedResidual(SCALE3_CHANNELS, SCALE3_CHANNELS,
                             stride=1, expand_ratio=EXPAND_RATIO),
        )

        # ========== LIGHTWEIGHT FPN ==========
        self.fpn = LightweightFPN(
            in_channels_s2=SCALE2_CHANNELS,
            in_channels_s3=SCALE3_CHANNELS,
            out_channels=FPN_CHANNELS,
        )

        # ========== DETECTION HEADS ==========
        # Head A: 8×8 grid (small/distant targets)
        self.head_small = SSDLiteHead(FPN_CHANNELS, num_anchors=num_anchors)
        # Head B: 4×4 grid (large/close targets)
        self.head_large = SSDLiteHead(FPN_CHANNELS, num_anchors=num_anchors)

        # ========== INTRUSION CLASSIFIER ==========
        self.classifier = IntrusionClassifier(
            in_channels=FPN_CHANNELS,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
        )

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
        Forward pass.

        Args:
            x: (B, 4, H, W) float tensor (3 RGB + 1 Thermal)

        Returns:
            dict with 'bbox_small', 'obj_small', 'bbox_large', 'obj_large', 'class_logits'
        """
        # 1. Dual Stems & Fusion
        x_rgb = x[:, :3, :, :]
        x_thermal = x[:, 3:, :, :]
        
        feat_rgb = self.rgb_stem(x_rgb)
        feat_thermal = self.thermal_stem(x_thermal)
        
        # Concat along channel dimension: (B, STEM_CHANNELS, H/2, W/2)
        feat_fused = torch.cat([feat_rgb, feat_thermal], dim=1)

        # 2. Backbone
        s1 = self.scale1(feat_fused)  # (B, C1, H/4, W/4)
        s2 = self.scale2(s1)          # (B, C2, H/8, W/8)
        s3 = self.scale3(s2)          # (B, C3, H/16, W/16)

        # FPN fusion
        p2, p3 = self.fpn(s2, s3)

        # Detection heads
        bbox_small, obj_small = self.head_small(p2)
        bbox_large, obj_large = self.head_large(p3)

        # Classification (objectness-weighted attention)
        label = self.classifier(p2, p3, obj_small, obj_large)

        return {
            'bbox_small': bbox_small,
            'obj_small': obj_small,
            'bbox_large': bbox_large,
            'obj_large': obj_large,
            'label': label,
        }


# ============================================================================
# 6. MODEL ANALYSIS UTILITIES
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model):
    """
    Estimate model size in different quantization formats.

    Returns:
        param_count, fp32_size_mb, int8_size_kb
    """
    param_count = count_parameters(model)
    fp32_mb = param_count * 4 / (1024 * 1024)
    int8_kb = param_count * 1 / 1024
    return param_count, fp32_mb, int8_kb


def estimate_peak_sram(model, input_size=None, batch_size=1):
    """
    Estimate peak SRAM usage during inference on ESP32-S3.

    Tracks maximum intermediate activation buffer size.
    """
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
    """Print comprehensive model analysis."""
    param_count, fp32_mb, int8_kb = estimate_model_size(model)
    sram = estimate_peak_sram(model)

    print(f"[OK] MicroGhost-Thermal:")
    print(f"   Parameters: {param_count:,}")
    print(f"   Est. Size (FP32): {param_count * 4 / 1024:,.1f} KB")
    print(f"   Est. Size (FP16): {param_count * 2 / 1024:,.1f} KB")
    print(f"  Target FP16:       <{ESP32_S3['target_model_fp16_kb']} KB "
          f"{'OK' if int8_kb * 2 < ESP32_S3['target_model_fp16_kb'] else 'FAIL'}")
    print()
    print(f"  Input buffer:      {sram['input_buffer_kb']:>10.1f} KB")
    print(f"  Peak act (FP32):   {sram['peak_activation_fp32_kb']:>10.1f} KB")
    print(f"  Peak act (INT8):   {sram['peak_activation_int8_kb']:>10.1f} KB")
    print(f"  Total arena INT8:  {sram['total_arena_int8_kb']:>10.1f} KB")
    print(f"  Fits ESP32-S3:     {'OK' if sram['fits_esp32_s3'] else 'FAIL'} "
          f"(limit: {ESP32_S3['max_arena_sram_kb']}KB)")

    # Layer-by-layer breakdown
    print(f"\n  {'Layer':<30} {'Params':>12} {'Size (KB)':>10}")
    print("  " + "-" * 54)

    components = [
        ('Stem', model.rgb_stem),
        ('Scale 1 (Ghost)', model.scale1),
        ('Scale 2 (InvRes)', model.scale2),
        ('Scale 3 (InvRes)', model.scale3),
        ('FPN', model.fpn),
        ('Head Small (8×8)', model.head_small),
        ('Head Large (4×4)', model.head_large),
        ('Classifier', model.classifier),
    ]

    total = 0
    for name, module in components:
        params = sum(p.numel() for p in module.parameters())
        kb = params * 4 / 1024
        total += params
        print(f"  {name:<30} {params:>12,} {kb:>8.1f} KB")

    print("  " + "-" * 54)
    print(f"  {'TOTAL':<30} {total:>12,} {total * 4 / 1024:>8.1f} KB")


# ============================================================================
# TEST
# ============================================================================

if __name__ == '__main__':
    print("Model Module — Self Test")
    print("-" * 40)

    # Create model
    model = MicroGhostThermal()

    # Print analysis
    print_model_analysis(model)

    # Test forward pass
    model.eval()
    h, w = INPUT_SIZE if isinstance(INPUT_SIZE, tuple) else (INPUT_SIZE, INPUT_SIZE)
    dummy = torch.randn(1, INPUT_CHANNELS, h, w)
    with torch.no_grad():
        outputs = model(dummy)

    print(f"\n[OK] Forward Pass Success! (input: {dummy.shape}):")
    for key, val in outputs.items():
        print(f"    {key}: {val.shape}")

    print("\n[OK] Model test passed!")
