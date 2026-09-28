"""
RUL Prediction Models for Bearing Prognostics

Architecture:
- Teacher Model: MAE-Base/Large (768/1024-dim), EfficientNet-B3 (1536-dim),
                 CLIP ViT-B/32 (512-dim) + RUL regression head
- Student Model: TinyViT (128-dim), EfficientNet-B0 (→128-dim),
                 MobileNet-V3-Small (→128-dim) + Multi-scale feature aligner + RUL head

Phase 1: Train teacher independently with frozen encoder
Phase 2: Train student with knowledge distillation from frozen teacher
Phase 3: K-window temporal aggregation + cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTMAEModel, ViTMAEConfig, ViTModel, ViTConfig
import torchvision.models as tv_models


class RULTeacherModel(nn.Module):
    """
    Teacher model for RUL prediction.

    Supported backbones (--teacher_vlm_type):
      mae_base       : MAE ViT-Base  (768-dim,  frozen ViT encoder)
      mae_large      : MAE ViT-Large (1024-dim, frozen ViT encoder)
      efficientnet_b3: EfficientNet-B3 ImageNet pretrained (1536-dim, frozen CNN)
      clip_vit_b32   : CLIP ViT-B/32 vision encoder (512-dim, frozen ViT encoder)

    All backbones expose the same forward() signature:
        (images, return_patch_tokens=False)
        → (rul_pred, cls_features)            [always]
        → (rul_pred, cls_features, patches)   [if return_patch_tokens=True]

    CNN backbones (efficientnet_b3) return patches=None because they have no
    patch token sequence; Phase-3 cross-modal attention degrades gracefully to
    the Phase-2 vision-only path in this case.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, args):
        super(RULTeacherModel, self).__init__()
        self.args = args
        self.device = self._acquire_device()

        teacher_vlm_type = getattr(args, 'teacher_vlm_type', 'mae_base')
        finetune_vlm     = getattr(args, 'finetune_vlm', False)
        dropout          = getattr(args, 'dropout', 0.1)

        self.teacher_vlm_type = teacher_vlm_type

        if teacher_vlm_type in ('mae_base', 'mae_large', 'mae_huge'):
            self._init_mae(args, teacher_vlm_type, finetune_vlm, dropout)
        elif teacher_vlm_type == 'efficientnet_b3':
            self._init_efficientnet_b3(finetune_vlm, dropout)
        elif teacher_vlm_type == 'clip_vit_b32':
            self._init_clip_vit_b32(finetune_vlm, dropout)
        else:
            print(f"Warning: Unknown teacher_vlm_type '{teacher_vlm_type}', falling back to mae_base")
            self._init_mae(args, 'mae_base', finetune_vlm, dropout)

        print(f"Teacher Model initialized: {teacher_vlm_type} (hidden={self.hidden_size}) + RUL Head")
        self._print_trainable_params()

    # ------------------------------------------------------------------
    # Backbone initialisers
    # ------------------------------------------------------------------

    def _init_mae(self, args, teacher_vlm_type, finetune_vlm, dropout):
        """MAE ViT-Base or ViT-Large encoder."""
        mae_size_map = {
            'mae_base':  {'path': 'facebook/vit-mae-base',  'hidden_size': 768},
            'mae_large': {'path': 'facebook/vit-mae-large', 'hidden_size': 1024},
            'mae_huge':  {'path': 'facebook/vit-mae-huge',  'hidden_size': 1280},
        }
        cfg = mae_size_map[teacher_vlm_type]
        pretrained_path = getattr(args, 'mae_pretrained_path', cfg['path'])
        print(f"Loading {teacher_vlm_type} from: {pretrained_path}")
        try:
            self.mae = ViTMAEModel.from_pretrained(pretrained_path)
            print(f"Successfully loaded pretrained {teacher_vlm_type}")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}. Using random weights.")
            self.mae = ViTMAEModel(ViTMAEConfig(hidden_size=cfg['hidden_size']))

        self.hidden_size   = self.mae.config.hidden_size
        self.use_cls_token = getattr(args, 'use_cls_token', True)
        self.backbone_type = 'vit'

        if not finetune_vlm:
            for p in self.mae.parameters():
                p.requires_grad = False
            print(f"MAE encoder frozen")
        else:
            print(f"MAE encoder trainable")

        self.rul_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )

    def _init_efficientnet_b3(self, finetune_vlm, dropout):
        """EfficientNet-B3 ImageNet pretrained — CNN teacher, hidden_size=1536."""
        try:
            backbone = tv_models.efficientnet_b3(
                weights=tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1
            )
            print("efficientnet_b3: loaded ImageNet pretrained weights")
        except Exception as e:
            print(f"efficientnet_b3: failed to load pretrained ({e}), random init")
            backbone = tv_models.efficientnet_b3(weights=None)

        feat_dim = backbone.classifier[1].in_features  # 1536
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        if not finetune_vlm:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("EfficientNet-B3 backbone frozen")
        else:
            print("EfficientNet-B3 backbone trainable")

        self.hidden_size   = feat_dim
        self.backbone_type = 'cnn'

        self.rul_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )

    def _init_clip_vit_b32(self, finetune_vlm, dropout):
        """CLIP ViT-B/32 vision encoder — hidden_size=512."""
        from transformers import CLIPVisionModel
        try:
            self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            print("clip_vit_b32: loaded pretrained CLIP vision encoder")
        except Exception as e:
            print(f"clip_vit_b32: failed to load pretrained ({e}), random init")
            from transformers import CLIPVisionConfig
            self.backbone = CLIPVisionModel(CLIPVisionConfig())

        self.hidden_size   = self.backbone.config.hidden_size  # 768 for ViT-B/32
        self.backbone_type = 'vit'

        if not finetune_vlm:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("CLIP vision encoder frozen")
        else:
            print("CLIP vision encoder trainable")

        self.rul_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.args.gpu}')
        return torch.device('cpu')

    def _print_trainable_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Teacher Model - Total params: {total:,}, Trainable: {trainable:,}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, return_patch_tokens=False):
        """
        Args:
            images: [B, 2, 224, 224] PSDI images
            return_patch_tokens: return spatial tokens for Phase-3 attention

        Returns:
            rul_pred     : [B, 1]
            cls_features : [B, hidden_size]
            patch_tokens : [B, N, hidden_size] or None  (only when return_patch_tokens=True)
        """
        images = _to_3ch(images)

        vlm = self.teacher_vlm_type
        if vlm in ('mae_base', 'mae_large', 'mae_huge'):
            return self._forward_mae(images, return_patch_tokens)
        elif vlm == 'efficientnet_b3':
            return self._forward_cnn(images, return_patch_tokens)
        elif vlm == 'clip_vit_b32':
            return self._forward_clip(images, return_patch_tokens)
        else:
            return self._forward_mae(images, return_patch_tokens)

    def _forward_mae(self, images, return_patch_tokens):
        outputs      = self.mae(images)
        hidden       = outputs.last_hidden_state             # [B, 1+N, D]
        if self.use_cls_token:
            cls_features = hidden[:, 0, :]                  # [B, D]
        else:
            cls_features = hidden.mean(dim=1)
        rul_pred = self.rul_head(cls_features)
        if return_patch_tokens:
            return rul_pred, cls_features, hidden[:, 1:, :]  # [B, N, D]
        return rul_pred, cls_features

    def _forward_cnn(self, images, return_patch_tokens):
        cls_features = self.backbone(images)                 # [B, 1536]
        rul_pred     = self.rul_head(cls_features)
        if return_patch_tokens:
            return rul_pred, cls_features, None              # CNN has no patch tokens
        return rul_pred, cls_features

    def _forward_clip(self, images, return_patch_tokens):
        outputs      = self.backbone(pixel_values=images)
        hidden       = outputs.last_hidden_state             # [B, 1+N, D]
        cls_features = hidden[:, 0, :]                       # [B, D]
        rul_pred     = self.rul_head(cls_features)
        if return_patch_tokens:
            return rul_pred, cls_features, hidden[:, 1:, :]  # [B, N, D]
        return rul_pred, cls_features


class MultiScaleFeatureAligner(nn.Module):
    """
    Multi-scale feature alignment for projecting student features to teacher dimension.

    Uses multiple projection paths at different scales and combines them with learnable weights.
    This helps bridge the dimension gap between small student and large teacher.

    Args:
        student_dim: student feature dimension (e.g., 128)
        teacher_dim: teacher feature dimension (e.g., 768)
        num_scales: number of projection scales (default 4)
        dropout: dropout rate (default 0.1)
    """
    def __init__(self, student_dim, teacher_dim, num_scales=4, dropout=0.1):
        super(MultiScaleFeatureAligner, self).__init__()

        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.num_scales = num_scales

        # Main direct projection
        self.main_proj = nn.Sequential(
            nn.Linear(student_dim, teacher_dim),
            nn.LayerNorm(teacher_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-scale projections with different hidden dimensions
        self.scale_projections = nn.ModuleList()
        for i in range(num_scales):
            # Hidden dimension decreases with scale
            hidden_dim = max(student_dim, teacher_dim) // (2 ** i)
            hidden_dim = max(hidden_dim, 64)  # Minimum hidden dimension

            proj = nn.Sequential(
                nn.Linear(student_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, teacher_dim),
                nn.LayerNorm(teacher_dim)
            )
            self.scale_projections.append(proj)

        # Learnable weights for combining scales
        # Initialize uniformly
        self.scale_weights = nn.Parameter(torch.ones(num_scales + 1) / (num_scales + 1))

        print(f"MultiScaleFeatureAligner: {student_dim} → {teacher_dim}, {num_scales} scales")

    def forward(self, student_features):
        """
        Project student features to teacher dimension.

        Args:
            student_features: [B, student_dim]

        Returns:
            aligned_features: [B, teacher_dim]
        """
        # Main projection
        main_out = self.main_proj(student_features)  # [B, teacher_dim]

        # Multi-scale projections
        scale_outs = [proj(student_features) for proj in self.scale_projections]

        # Normalize weights with softmax
        normalized_weights = F.softmax(self.scale_weights, dim=0)

        # Weighted combination
        aligned_features = normalized_weights[0] * main_out
        for i, scale_out in enumerate(scale_outs):
            aligned_features += normalized_weights[i + 1] * scale_out

        return aligned_features


class TemporalConvAggregator(nn.Module):
    """
    Temporal aggregation over K timesteps using 1D convolution with residual connections.

    Input:  [B, K, feature_dim]
    Output: [B, feature_dim]

    When K=1, AdaptiveAvgPool1d(1) acts as identity → backward compatible.
    """

    def __init__(self, feature_dim, num_layers=2, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """x: [B, K, D] → [B, D]"""
        x = x.permute(0, 2, 1)  # [B, D, K]
        for conv in self.conv_layers:
            x = x + conv(x)  # residual
        x = self.pool(x)  # [B, D, 1]
        return x.squeeze(-1)  # [B, D]


class RULStudentModel(nn.Module):
    """
    Student model for RUL prediction.

    Supported backbones (--student_vision_type):
      tiny_vit       : Custom 4-layer ViT (128-dim natively)
      efficientnet_b0: EfficientNet-B0 ImageNet pretrained (1280→128 via feat_proj)
      mobilenet_v3   : MobileNet-V3-Small ImageNet pretrained (576→128 via feat_proj)

    All backbones output student_hidden_size=128 features before the aligner.
    CNN backbones (efficientnet_b0, mobilenet_v3) return patch_tokens=None;
    Phase-3 cross-modal attention falls back to Phase-2 vision-only path.

    Args:
        args: Configuration object with:
            - student_vision_type: backbone selection (default 'tiny_vit')
            - student_hidden_size: projected feature dim (default 128)
            - teacher_hidden_size: teacher feature dim for aligner (default 768)
            - num_alignment_scales: scales for feature aligner (default 4)
            - feature_alignment_dropout: dropout for feature aligner (default 0.1)
    """
    def __init__(self, args):
        super(RULStudentModel, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.rul_phase  = getattr(args, 'rul_phase', 1)
        self.d_model    = getattr(args, 'd_model', 128)

        # Student configuration
        self.student_hidden_size   = getattr(args, 'student_hidden_size', 128)
        self.teacher_hidden_size   = getattr(args, 'teacher_hidden_size', 768)
        self.phase3_qkv_mode       = getattr(args, 'phase3_qkv_mode', 'vision_q_phase_kv')
        self.phase3_residual_fusion = getattr(args, 'phase3_residual_fusion', False)
        self.student_vision_type   = getattr(args, 'student_vision_type', 'tiny_vit')

        dropout    = getattr(args, 'dropout', 0.1)
        finetune   = getattr(args, 'finetune_vlm', False)

        # --- Vision backbone ---
        if self.student_vision_type == 'tiny_vit':
            self._init_tiny_vit(dropout)
        elif self.student_vision_type == 'efficientnet_b0':
            self._init_efficientnet_b0(finetune, dropout)
        elif self.student_vision_type == 'mobilenet_v3':
            self._init_mobilenet_v3(finetune, dropout)
        else:
            print(f"Warning: Unknown student_vision_type '{self.student_vision_type}', "
                  f"falling back to tiny_vit")
            self.student_vision_type = 'tiny_vit'
            self._init_tiny_vit(dropout)

        # Multi-scale feature aligner (student_hidden_size → teacher_hidden_size)
        num_scales = getattr(args, 'num_alignment_scales', 4)
        align_dropout = getattr(args, 'feature_alignment_dropout', 0.1)

        self.feature_aligner = MultiScaleFeatureAligner(
            student_dim=self.student_hidden_size,
            teacher_dim=self.teacher_hidden_size,
            num_scales=num_scales,
            dropout=align_dropout,
        )

        # Phase 3: phase-space embedding and cross-modal attention
        self.phase_input_dim = 2 * 2560
        self.num_phase_tokens = 160
        self.phase_token_dim = self.d_model
        self.use_phase3 = self.rul_phase >= 3
        if self.use_phase3:
            self.phase_projection = nn.Linear(
                self.phase_input_dim,
                self.num_phase_tokens * self.phase_token_dim
            )
            self.vision_patch_projection = nn.Linear(
                self.student_hidden_size, self.phase_token_dim
            )
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=self.phase_token_dim,
                num_heads=4,
                dropout=getattr(args, 'dropout', 0.1),
                batch_first=True
            )
            self.phase_to_teacher = nn.Sequential(
                nn.Linear(self.phase_token_dim, self.teacher_hidden_size),
                nn.LayerNorm(self.teacher_hidden_size),
                nn.GELU()
            )
            self.attention_dropout = nn.Dropout(getattr(args, 'dropout', 0.1))
            if self.phase3_residual_fusion:
                alpha_init = float(getattr(args, 'phase3_fusion_alpha_init', 0.7))
                alpha_init = min(max(alpha_init, 1e-4), 1 - 1e-4)
                alpha_logit = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
                self.fusion_alpha_logit = nn.Parameter(alpha_logit)
                print(f"Phase3 residual fusion enabled (alpha_init={alpha_init:.3f})")

        # RUL regression head (operates on teacher dimension)
        self.rul_head = nn.Sequential(
            nn.Linear(self.teacher_hidden_size, self.teacher_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.teacher_hidden_size // 2, 1)
        )

        # K-Window temporal aggregation
        self.K = getattr(args, 'k_window_size', 1)
        if self.K > 1:
            self.temporal_aggregator = TemporalConvAggregator(
                feature_dim=self.teacher_hidden_size,
                num_layers=getattr(args, 'temporal_conv_layers', 2),
                kernel_size=getattr(args, 'temporal_conv_kernel', 3),
                dropout=dropout,
            )
            print(f"K-Window temporal aggregation enabled (K={self.K})")

        print(f"Student Model initialized: {self.student_vision_type} "
              f"(hidden={self.student_hidden_size}) + Feature Aligner + RUL Head")
        self._print_trainable_params()

    # ------------------------------------------------------------------
    # Backbone initialisers
    # ------------------------------------------------------------------

    def _init_tiny_vit(self, dropout):
        """4-layer custom ViT with 128 hidden — original TinyViT student."""
        vit_config = ViTConfig(
            hidden_size=self.student_hidden_size,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=self.student_hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            image_size=224,
            patch_size=16,
            num_channels=3,
        )
        self.vit = ViTModel(vit_config)
        self.backbone_type = 'vit'
        print(f"TinyViT student initialised (hidden={self.student_hidden_size}, 4 layers)")

    def _init_efficientnet_b0(self, finetune, dropout):
        """EfficientNet-B0 ImageNet pretrained → Linear projection to 128."""
        try:
            backbone = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            print("Student EfficientNet-B0: loaded ImageNet pretrained weights")
        except Exception as e:
            print(f"Student EfficientNet-B0: failed to load pretrained ({e}), random init")
            backbone = tv_models.efficientnet_b0(weights=None)

        raw_feat_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.backbone_type = 'cnn'

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Student EfficientNet-B0 backbone frozen")
        else:
            print("Student EfficientNet-B0 backbone trainable")

        # Project raw features → student_hidden_size (128)
        self.feat_proj = nn.Sequential(
            nn.Linear(raw_feat_dim, self.student_hidden_size),
            nn.GELU(),
        )
        print(f"EfficientNet-B0 student: {raw_feat_dim} → {self.student_hidden_size} via feat_proj")

    def _init_mobilenet_v3(self, finetune, dropout):
        """MobileNet-V3-Small ImageNet pretrained → Linear projection to 128."""
        try:
            backbone = tv_models.mobilenet_v3_small(
                weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            print("Student MobileNet-V3-Small: loaded ImageNet pretrained weights")
        except Exception as e:
            print(f"Student MobileNet-V3-Small: failed to load pretrained ({e}), random init")
            backbone = tv_models.mobilenet_v3_small(weights=None)

        raw_feat_dim = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.backbone_type = 'cnn'

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("Student MobileNet-V3-Small backbone frozen")
        else:
            print("Student MobileNet-V3-Small backbone trainable")

        # Project raw features → student_hidden_size (128)
        self.feat_proj = nn.Sequential(
            nn.Linear(raw_feat_dim, self.student_hidden_size),
            nn.GELU(),
        )
        print(f"MobileNet-V3-Small student: {raw_feat_dim} → {self.student_hidden_size} via feat_proj")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            return torch.device(f'cuda:{self.args.gpu}')
        return torch.device('cpu')

    def _print_trainable_params(self):
        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Student Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")

    def forward(self, images, P=None):
        """
        Forward pass for simple inference.

        Args:
            images: [B, 2, 224, 224] PSDI images

        Returns:
            rul_pred: [B, 1] RUL prediction
        """
        rul_pred, _, _ = self.forward_with_features(images, P=P)
        return rul_pred

    def _phase3_forward(self, patch_tokens, P, return_attention=False):
        """
        Phase 3 forward with cross-modal attention.

        Args:
            patch_tokens: [B, 196, student_hidden]
            P: [B, 2, 2560] or [B, 5120]
            return_attention: whether to return attention map

        Returns:
            phase3_teacher_features: [B, teacher_hidden]
            attention_map:
              - [B, 196, 160] for vision_q_phase_kv
              - [B, 160, 196] for phase_q_vision_kv
              - None if return_attention=False
        """
        batch_size = patch_tokens.size(0)

        if P.dim() == 3:
            phase_flat = P.reshape(batch_size, -1)  # [B, 5120]
        elif P.dim() == 2:
            phase_flat = P
        else:
            raise ValueError(f"Unexpected Phase Space shape: {P.shape}")

        phase_tokens = self.phase_projection(phase_flat).view(
            batch_size, self.num_phase_tokens, self.phase_token_dim
        )  # [B, 160, d_model]
        vision_tokens = self.vision_patch_projection(patch_tokens)  # [B, 196, d_model]

        if self.phase3_qkv_mode == 'vision_q_phase_kv':
            # Legacy best-performing setting:
            # Q=vision, K/V=phase.
            attended_tokens, attn_weights = self.cross_modal_attention(
                query=vision_tokens,
                key=phase_tokens,
                value=phase_tokens,
                need_weights=return_attention
            )
            attended_tokens = self.attention_dropout(attended_tokens)
            if return_attention and attn_weights is not None:
                patch_weights = attn_weights.mean(dim=-1)  # [B, 196]
            else:
                patch_weights = torch.ones(
                    batch_size, vision_tokens.size(1), device=vision_tokens.device
                )
            patch_weights = patch_weights / (patch_weights.sum(dim=1, keepdim=True) + 1e-8)
            pooled_features = (attended_tokens * patch_weights.unsqueeze(-1)).sum(dim=1)
        elif self.phase3_qkv_mode == 'phase_q_vision_kv':
            # Alternative setting for ablation:
            # Q=phase, K/V=vision.
            attended_tokens, attn_weights = self.cross_modal_attention(
                query=phase_tokens,
                key=vision_tokens,
                value=vision_tokens,
                need_weights=return_attention
            )
            attended_tokens = self.attention_dropout(attended_tokens)
            if return_attention and attn_weights is not None:
                phase_weights = attn_weights.mean(dim=-1)  # [B, 160]
                phase_weights = phase_weights / (phase_weights.sum(dim=1, keepdim=True) + 1e-8)
                pooled_features = (attended_tokens * phase_weights.unsqueeze(-1)).sum(dim=1)
            else:
                pooled_features = attended_tokens.mean(dim=1)
        else:
            raise ValueError(f"Unsupported phase3_qkv_mode: {self.phase3_qkv_mode}")

        phase3_teacher_features = self.phase_to_teacher(pooled_features)
        return phase3_teacher_features, attn_weights

    def forward_with_features(self, images, P=None, teacher_vision_feat=None,
                             teacher_patch_tokens=None, return_attention=False):
        """
        Forward pass with intermediate features (for distillation).

        NOTE: Returns RAW student features (not aligned).
        Feature alignment is handled by distill.py for consistency with time series forecasting.

        Args:
            images: [B, 2, 224, 224] or [B, K, 2, 224, 224] PSDI images
            P: [B, 2, 2560] or [B, K, 2, 2560] Phase Space features (used in Phase 3)
            teacher_vision_feat: [B, teacher_hidden] teacher CLS features (not used, for compatibility)
            teacher_patch_tokens: [B, 196, teacher_hidden] teacher patches (not used in Phase 2)
            return_attention: whether to return attention maps (False in Phase 2)

        Returns:
            rul_pred: [B, 1] RUL prediction
            student_features: [B, student_hidden_size] RAW student features (last timestep, NOT aligned)
            attention: attention map or None
        """
        # K-Window path: images is [B, K, C, H, W]
        if images.dim() == 5 and self.K > 1:
            return self._forward_kwindow(images, P, return_attention)

        # Single-image path: images is [B, C, H, W]
        images = _to_3ch(images)
        student_features, patch_tokens = self._encode_single(images)

        # Phase 3: cross-modal attention when patch_tokens available and P provided.
        if self.use_phase3 and P is not None and patch_tokens is not None:
            phase3_features, attention = self._phase3_forward(
                patch_tokens, P, return_attention=return_attention
            )
            if self.phase3_residual_fusion:
                vision_features = self.feature_aligner(student_features)
                alpha = torch.sigmoid(self.fusion_alpha_logit)
                aligned_features = alpha * phase3_features + (1.0 - alpha) * vision_features
            else:
                aligned_features = phase3_features
        else:
            # Phase 1/2 path (also fallback for CNN students without patch_tokens).
            aligned_features = self.feature_aligner(student_features)
            attention = None

        rul_pred = self.rul_head(aligned_features)
        return rul_pred, student_features, attention

    def _encode_single(self, images_3ch):
        """
        Encode a batch of 3-channel images through the student backbone.

        Returns:
            student_features : [B, student_hidden_size]  — CLS / pooled feature
            patch_tokens     : [B, N, student_hidden_size] or None (CNN backbones)
        """
        if self.backbone_type == 'vit':
            outputs      = self.vit(images_3ch)
            hidden       = outputs.last_hidden_state          # [B, 1+N, D]
            cls_feat     = hidden[:, 0, :]                    # [B, D]
            patch_tokens = hidden[:, 1:, :]                   # [B, N, D]
            return cls_feat, patch_tokens
        else:  # CNN: efficientnet_b0, mobilenet_v3
            raw_feat     = self.backbone(images_3ch)           # [B, raw_dim]
            cls_feat     = self.feat_proj(raw_feat)            # [B, 128]
            return cls_feat, None                              # no patch tokens for CNN

    def _forward_kwindow(self, images, P, return_attention=False):
        """
        K-Window forward: process K consecutive images through shared backbone,
        then aggregate temporally.

        Args:
            images: [B, K, 2, 224, 224]
            P: [B, K, 2, 2560] or None
            return_attention: whether to return attention maps

        Returns:
            rul_pred: [B, 1]
            student_features_last: [B, student_hidden_size] (last timestep, raw)
            attention: averaged attention map or None
        """
        batch_size, K = images.shape[0], images.shape[1]

        # Flatten [B, K, 2, H, W] → [B*K, 3, H, W]
        images_flat = _to_3ch(images.view(batch_size * K, *images.shape[2:]))

        # Shared backbone: encode all K frames at once
        cls_flat, patch_flat = self._encode_single(images_flat)

        # Reshape cls to [B, K, student_hidden]
        cls_K = cls_flat.view(batch_size, K, -1)

        # Phase 3 path (only when ViT backbone provides patch tokens)
        if self.use_phase3 and P is not None and patch_flat is not None:
            patch_K = patch_flat.view(batch_size, K, patch_flat.size(1), -1)  # [B,K,N,D]
            features_list, attn_list = [], []
            for k in range(K):
                feat_k, attn_k = self._phase3_forward(
                    patch_K[:, k], P[:, k], return_attention=return_attention
                )
                features_list.append(feat_k)
                if return_attention and attn_k is not None:
                    attn_list.append(attn_k)

            features_K = torch.stack(features_list, dim=1)   # [B, K, teacher_hidden]
            aggregated  = self.temporal_aggregator(features_K)

            if self.phase3_residual_fusion:
                vision_features = self.feature_aligner(cls_K[:, -1])
                alpha = torch.sigmoid(self.fusion_alpha_logit)
                aligned_features = alpha * aggregated + (1.0 - alpha) * vision_features
            else:
                aligned_features = aggregated

            attention = torch.stack(attn_list).mean(dim=0) if attn_list else None
        else:
            # Phase 1/2 path (also CNN students without patch tokens)
            aligned_list = [self.feature_aligner(cls_K[:, k]) for k in range(K)]
            aligned_K    = torch.stack(aligned_list, dim=1)  # [B, K, teacher_hidden]
            aligned_features = self.temporal_aggregator(aligned_K)
            attention = None

        rul_pred = self.rul_head(aligned_features)
        return rul_pred, cls_K[:, -1], attention

    def get_phase3_fusion_alpha(self):
        """Return current residual fusion alpha if enabled."""
        if self.use_phase3 and self.phase3_residual_fusion:
            return torch.sigmoid(self.fusion_alpha_logit).item()
        return None


def _make_rul_head(in_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """Shared RUL regression head used by all CNN baselines."""
    return nn.Sequential(
        nn.Linear(in_dim, in_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(in_dim // 2, 1),
    )


def _to_3ch(images: torch.Tensor) -> torch.Tensor:
    """Convert 2-channel PSDI images to 3 channels by repeating ch0."""
    if images.size(1) == 2:
        return torch.cat([images, images[:, :1]], dim=1)
    return images


class SimpleCNNRULModel(nn.Module):
    """
    Lightweight 4-block CNN trained from scratch on PSDI images.

    Architecture:
        Conv(3→32) → BN → ReLU → MaxPool    [112×112]
        Conv(32→64) → BN → ReLU → MaxPool   [56×56]
        Conv(64→128) → BN → ReLU → MaxPool  [28×28]
        Conv(128→256) → BN → ReLU → AAP(1)  [B, 256]
        RUL head: Linear(256→128) → GELU → Dropout → Linear(128→1)

    Args:
        args: config object (uses args.dropout, args.use_gpu, args.gpu)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        dropout = getattr(args, 'dropout', 0.1)

        def _block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            _block(3, 32),    # 112×112
            _block(32, 64),   # 56×56
            _block(64, 128),  # 28×28
            _block(128, 256, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.rul_head = _make_rul_head(256, dropout)
        print("SimpleCNNRULModel initialized (4 conv blocks, scratch)")
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"SimpleCNN - Total: {total:,}, Trainable: {trainable:,}")

    def forward(self, images):
        images = _to_3ch(images)
        features = self.encoder(images).flatten(1)  # [B, 256]
        rul_pred = self.rul_head(features)           # [B, 1]
        return rul_pred, features


class VGGRULModel(nn.Module):
    """
    VGG-style deep CNN trained from scratch on PSDI images.

    Architecture:
        Block1: Conv(3→64)×2 → MaxPool   [112×112]
        Block2: Conv(64→128)×2 → MaxPool [56×56]
        Block3: Conv(128→256)×3 → MaxPool [28×28]
        Block4: Conv(256→512)×3 → AAP(1) [B, 512]
        RUL head: Linear(512→256) → GELU → Dropout → Linear(256→1)

    Args:
        args: config object (uses args.dropout)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        dropout = getattr(args, 'dropout', 0.1)

        def _vgg_block(in_ch, out_ch, num_convs, pool=True):
            layers = []
            for i in range(num_convs):
                layers += [
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.encoder = nn.Sequential(
            _vgg_block(3, 64, 2),    # 112×112
            _vgg_block(64, 128, 2),  # 56×56
            _vgg_block(128, 256, 3), # 28×28
            _vgg_block(256, 512, 3, pool=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.rul_head = _make_rul_head(512, dropout)
        print("VGGRULModel initialized (VGG-style, scratch)")
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"VGG-style CNN - Total: {total:,}, Trainable: {trainable:,}")

    def forward(self, images):
        images = _to_3ch(images)
        features = self.encoder(images).flatten(1)  # [B, 512]
        rul_pred = self.rul_head(features)           # [B, 1]
        return rul_pred, features


class ResNetRULModel(nn.Module):
    """
    ResNet-18 or ResNet-34 backbone (ImageNet pretrained) for RUL prediction.

    Replaces the final FC layer with an identity to expose [B, 512] features,
    then adds a lightweight RUL head.

    Args:
        args: config object with:
            - cnn_type: 'resnet18' or 'resnet34'
            - finetune_cnn_backbone: whether to finetune backbone (default True)
            - dropout: dropout rate (default 0.1)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        cnn_type = getattr(args, 'cnn_type', 'resnet18')
        dropout = getattr(args, 'dropout', 0.1)
        finetune = getattr(args, 'finetune_cnn_backbone', True)

        if cnn_type == 'resnet34':
            try:
                backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
                print("resnet34: loaded ImageNet pretrained weights")
            except Exception as e:
                print(f"resnet34: failed to load pretrained weights ({e}), using random init")
                backbone = tv_models.resnet34(weights=None)
        else:
            try:
                backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
                print("resnet18: loaded ImageNet pretrained weights")
            except Exception as e:
                print(f"resnet18: failed to load pretrained weights ({e}), using random init")
                backbone = tv_models.resnet18(weights=None)

        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"{cnn_type} backbone frozen (finetune_cnn_backbone=False)")
        else:
            print(f"{cnn_type} backbone trainable (finetune_cnn_backbone=True)")

        self.rul_head = _make_rul_head(feat_dim, dropout)
        print(f"ResNetRULModel initialized ({cnn_type}, feat_dim={feat_dim})")
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"ResNet CNN - Total: {total:,}, Trainable: {trainable:,}")

    def forward(self, images):
        images = _to_3ch(images)
        features = self.backbone(images)  # [B, 512]
        rul_pred = self.rul_head(features)  # [B, 1]
        return rul_pred, features


class EfficientNetRULModel(nn.Module):
    """
    EfficientNet-B0 backbone (ImageNet pretrained) for RUL prediction.

    Replaces the final classifier with an identity to expose [B, 1280] features,
    then adds a lightweight RUL head.

    Args:
        args: config object with:
            - finetune_cnn_backbone: whether to finetune backbone (default True)
            - dropout: dropout rate (default 0.1)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        dropout = getattr(args, 'dropout', 0.1)
        finetune = getattr(args, 'finetune_cnn_backbone', True)

        try:
            backbone = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            print("efficientnet_b0: loaded ImageNet pretrained weights")
        except Exception as e:
            print(f"efficientnet_b0: failed to load pretrained weights ({e}), using random init")
            backbone = tv_models.efficientnet_b0(weights=None)

        feat_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        if not finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("EfficientNet-B0 backbone frozen (finetune_cnn_backbone=False)")
        else:
            print("EfficientNet-B0 backbone trainable (finetune_cnn_backbone=True)")

        self.rul_head = _make_rul_head(feat_dim, dropout)
        print(f"EfficientNetRULModel initialized (B0, feat_dim={feat_dim})")
        self._print_params()

    def _print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EfficientNet CNN - Total: {total:,}, Trainable: {trainable:,}")

    def forward(self, images):
        images = _to_3ch(images)
        features = self.backbone(images)  # [B, 1280]
        rul_pred = self.rul_head(features)  # [B, 1]
        return rul_pred, features


class Model(nn.Module):
    """
    Wrapper for RUL models (for compatibility with exp_rul.py).

    Automatically selects Teacher or Student model based on args.is_student.
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.is_student = getattr(args, 'is_student', False)
        cnn_type = getattr(args, 'cnn_type', None)

        # CNN baseline path — mutually exclusive with teacher/student
        _CNN_TYPES = ('simple', 'vgg', 'resnet18', 'resnet34', 'efficientnet_b0')
        if cnn_type in _CNN_TYPES:
            self.is_student = False
            self.is_cnn = True
            if cnn_type == 'simple':
                print("Building SimpleCNN baseline for RUL Prediction")
                self.model = SimpleCNNRULModel(args)
            elif cnn_type == 'vgg':
                print("Building VGG-style CNN baseline for RUL Prediction")
                self.model = VGGRULModel(args)
            elif cnn_type in ('resnet18', 'resnet34'):
                print(f"Building {cnn_type} CNN baseline for RUL Prediction")
                self.model = ResNetRULModel(args)
            elif cnn_type == 'efficientnet_b0':
                print("Building EfficientNet-B0 CNN baseline for RUL Prediction")
                self.model = EfficientNetRULModel(args)
        else:
            self.is_cnn = False
            if self.is_student:
                print("Building Student Model for RUL Prediction")
                self.model = RULStudentModel(args)
            else:
                print("Building Teacher Model for RUL Prediction")
                self.model = RULTeacherModel(args)

    def forward(self, images, P=None):
        """
        Forward pass.

        Args:
            images: [B, 2, 224, 224]
            P: [B, 2, 2560] (optional, not used in Phase 1 & 2)

        Returns:
            rul_pred: [B, 1]
        """
        if self.is_cnn:
            rul_pred, _ = self.model.forward(images)
            return rul_pred
        elif self.is_student:
            return self.model.forward(images, P=P)
        else:
            rul_pred, _ = self.model.forward(images)
            return rul_pred

    def forward_with_features(self, images, P=None, teacher_vision_feat=None,
                             teacher_patch_tokens=None, return_attention=False):
        """Forward with features (for distillation)"""
        if self.is_cnn:
            rul_pred, feat = self.model.forward(images)
            return rul_pred, feat, None
        elif self.is_student:
            return self.model.forward_with_features(
                images, P, teacher_vision_feat, teacher_patch_tokens, return_attention
            )
        else:
            # Teacher doesn't have this method in Phase 1
            rul_pred, cls_feat = self.model.forward(images)
            return rul_pred, cls_feat, None

    def get_phase3_fusion_alpha(self):
        """Expose student's fusion alpha for logging."""
        if self.is_student and hasattr(self.model, 'get_phase3_fusion_alpha'):
            return self.model.get_phase3_fusion_alpha()
        return None


def prepare_images_for_mae(images):
    """
    Prepare 2-channel PSDI images for MAE input.

    Args:
        images: [B, 2, H, W] 2-channel images

    Returns:
        images_3ch: [B, 3, H, W] 3-channel images
    """
    if images.size(1) == 2:
        # Repeat first channel to create 3 channels
        images_3ch = torch.cat([images, images[:, :1, :, :]], dim=1)
    elif images.size(1) == 3:
        images_3ch = images
    else:
        raise ValueError(f"Expected 2 or 3 channels, got {images.size(1)}")

    return images_3ch
