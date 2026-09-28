"""
Experiment class for bearing RUL prediction with 3-phase distillation.

Follows the same two-stage pattern as exp_long_term_forecasting.py:
  Stage 1 — train teacher independently on RUL task
  Stage 2 — freeze teacher, train student with distillation loss
"""
import json
import os
import time
import csv
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.model_profiler import profile_model, format_profile_line
from src.psdi_kd.distill import DistillationLoss
from src.psdi_kd.rul_model import RULTeacherModel

warnings.filterwarnings('ignore')


class Exp_RUL(Exp_Basic):
    def __init__(self, args):
        # Set phase BEFORE super().__init__() because _build_model() needs it
        self.phase = getattr(args, 'rul_phase', 1)
        self.use_distillation = self.phase >= 2

        # Now call parent init which will call _build_model()
        super(Exp_RUL, self).__init__(args)

        # Build teacher model for distillation (Phase 2+)
        if self.use_distillation:
            self.teacher_model = self._build_teacher_model()
            self.distill_loss = DistillationLoss(args)

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def _build_model(self):
        from src.psdi_kd.rul_model import Model

        # In Phase 2, build student model; in Phase 1, build teacher model
        if self.phase >= 2:
            self.args.is_student = True

        model = Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _build_teacher_model(self):
        """Build teacher model (MAE-base + RUL head)."""
        original_args = vars(self.args).copy()

        # Configure for teacher
        self.args.is_student = False
        # Note: finetune_vlm is controlled by args, default is False (freeze MAE)
        if hasattr(self.args, 'teacher_vlm_type'):
            self.args.vlm_type = self.args.teacher_vlm_type

        teacher = RULTeacherModel(self.args).float().to(self.device)

        # Restore original args
        for key, value in original_args.items():
            setattr(self.args, key, value)

        if self.args.use_multi_gpu and self.args.use_gpu:
            teacher = nn.DataParallel(teacher, device_ids=self.args.device_ids)

        return teacher

    def _freeze_teacher(self):
        """Completely freeze teacher model."""
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model frozen.")

    def _get_model_core(self):
        """Return bare model regardless of DataParallel wrapper."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_phase3_fusion_alpha(self):
        """Return current phase3 residual fusion alpha when enabled."""
        model_core = self._get_model_core()
        if hasattr(model_core, 'get_phase3_fusion_alpha'):
            return model_core.get_phase3_fusion_alpha()
        return None

    def _build_teacher_attention_target(self, student_attn, teacher_patch_tokens):
        """
        Build teacher attention target from teacher patch-token correlations.

        Args:
            student_attn: [B, Q, K] student attention logits/weights
            teacher_patch_tokens: [B, N, D] teacher patch tokens (N ~= 196)

        Returns:
            teacher_attn: [B, Q, K] teacher-derived attention target, or None.
        """
        if student_attn is None or teacher_patch_tokens is None:
            return None

        # Correlation logits over teacher patches: [B, N, N]
        patch_tokens = F.normalize(teacher_patch_tokens, dim=-1)
        scale = patch_tokens.size(-1) ** 0.5
        teacher_corr = torch.matmul(patch_tokens, patch_tokens.transpose(1, 2)) / scale

        # Resize correlation map to match student attention shape [Q, K].
        q_len, k_len = student_attn.size(1), student_attn.size(2)
        teacher_attn = F.interpolate(
            teacher_corr.unsqueeze(1),  # [B, 1, N, N]
            size=(q_len, k_len),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)  # [B, Q, K]
        return teacher_attn

    def _append_epoch_log(self, csv_path, row):
        """Append one epoch summary row to CSV."""
        fieldnames = [
            'epoch',
            'model_lr',
            'distill_lr',
            'train_total',
            'val_total',
            'feature_loss',
            'fcst_loss',
            'recon_loss',
            'att_loss',
            'deg_weight_mean',
            'feature_w',
            'fcst_w',
            'recon_w',
            'att_w',
            'temperature',
            'phase3_fusion_alpha',
        ]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Model profiling
    # ------------------------------------------------------------------

    def _save_model_profile(self, setting, folder_path):
        """
        Profile teacher and student forward passes, save to model_profile.json.
        Also returns the profile dict for embedding into metrics.txt.
        """
        device = self.device
        dummy = torch.zeros(1, 2, 224, 224, device=device)

        profile_data = {
            'setting':            setting,
            'dataset':            getattr(self.args, 'dataset_name', 'unknown'),
            'k_window_size':      getattr(self.args, 'k_window_size', 1),
            'temporal_conv_layers': getattr(self.args, 'temporal_conv_layers', 2),
        }

        # --- Student model ---
        try:
            student_core = self.model.module if hasattr(self.model, 'module') else self.model
            # For K-window, profile with single image (K=1 path) to isolate backbone cost
            p_student = profile_model(student_core, dummy, device)
            profile_data['student'] = {
                'vision_type':      getattr(self.args, 'student_vision_type', 'tiny_vit'),
                'hidden_size':      getattr(self.args, 'student_hidden_size', 128),
                'encoder_params_M': p_student['encoder_params_M'],
                'total_params_M':   p_student['total_params_M'],
                'mem_mib':          p_student['mem_mib'],
                'speed_s_per_iter': p_student['speed_s_per_iter'],
            }
        except Exception as e:
            print(f"[Profiler] Student profiling failed: {e}")
            profile_data['student'] = {}

        # --- Teacher model (only available in Phase 2+) ---
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            try:
                teacher_core = (self.teacher_model.module
                                if hasattr(self.teacher_model, 'module')
                                else self.teacher_model)
                p_teacher = profile_model(teacher_core, dummy, device)
                profile_data['teacher'] = {
                    'vlm_type':         getattr(self.args, 'teacher_vlm_type', 'mae_base'),
                    'hidden_size':      getattr(self.args, 'teacher_hidden_size', 768),
                    'encoder_params_M': p_teacher['encoder_params_M'],
                    'total_params_M':   p_teacher['total_params_M'],
                    'mem_mib':          p_teacher['mem_mib'],
                    'speed_s_per_iter': p_teacher['speed_s_per_iter'],
                }
            except Exception as e:
                print(f"[Profiler] Teacher profiling failed: {e}")
                profile_data['teacher'] = {}
        else:
            # Phase 1: teacher IS self.model
            profile_data['teacher'] = profile_data.get('student', {})
            profile_data.pop('student', None)

        os.makedirs(folder_path, exist_ok=True)
        json_path = os.path.join(folder_path, 'model_profile.json')
        with open(json_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        print(f"[Profiler] Saved to {json_path}")

        return profile_data

    # ------------------------------------------------------------------
    # Data & optimizers
    # ------------------------------------------------------------------
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_criterion(self):
        return nn.SmoothL1Loss()

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        if self.use_distillation:
            distill_params = self.distill_loss.get_learnable_parameters()
            if distill_params:
                distill_lr = self.args.learning_rate * getattr(
                    self.args, 'distill_lr_ratio', 0.1)
                distill_optim = optim.Adam(distill_params, lr=distill_lr)
                return model_optim, distill_optim

        return model_optim

    # ------------------------------------------------------------------
    # Training router
    # ------------------------------------------------------------------
    def train(self, setting):
        if self.phase == 1:
            return self._train_standard(setting)
        if self.phase == 3:
            return self._train_phase3(setting)
        else:
            return self._train_two_stage(setting)

    def _load_checkpoint_to_model(self, model, ckpt_path, strict=True):
        """Load checkpoint with DataParallel compatibility."""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        target = model.module if isinstance(model, nn.DataParallel) else model

        try:
            target.load_state_dict(checkpoint, strict=strict)
            print(f"Loaded checkpoint: {ckpt_path}")
            return
        except RuntimeError:
            pass

        remapped = {}
        add_module_prefix = not all(k.startswith('module.') for k in target.state_dict().keys())
        for key, value in checkpoint.items():
            if add_module_prefix:
                remapped[key.replace('module.', '', 1)] = value
            else:
                remapped[f"module.{key}"] = value

        target.load_state_dict(remapped, strict=strict)
        print(f"Loaded checkpoint with key remap: {ckpt_path}")

    def _train_phase3(self, setting):
        """
        Phase 3: cross-modal training with X + P.
        Supports:
          - from scratch (teacher stage + student stage, both in this run)
          - resume from Phase 2 checkpoints
        """
        print("=== Phase 3: Cross-modal distillation refinement ===")

        phase3_from_scratch = getattr(self.args, 'phase3_from_scratch', False)
        skip_teacher_stage = getattr(self.args, 'skip_teacher_stage_in_phase3', True)
        phase2_ckpt_path = getattr(self.args, 'phase2_ckpt_path', '')
        phase2_teacher_ckpt_path = getattr(self.args, 'phase2_teacher_ckpt_path', '')

        if phase3_from_scratch:
            print("Phase 3 mode: train from scratch (no Phase 2 checkpoint).")
            teacher_setting = setting + "_teacher"
            self._train_teacher(teacher_setting)
            self._freeze_teacher()
            # Student stage uses X + P because self.phase == 3.
            return self._train_with_distillation(setting)

        if not phase2_ckpt_path:
            raise ValueError(
                "Phase 3 resume mode requires --phase2_ckpt_path. "
                "Or set --phase3_from_scratch true."
            )

        self._load_checkpoint_to_model(self.model, phase2_ckpt_path, strict=False)

        if skip_teacher_stage:
            if not phase2_teacher_ckpt_path:
                # Default: infer teacher path from phase2 checkpoint directory.
                # e.g., checkpoints/xxx/checkpoint.pth -> checkpoints/xxx_teacher/checkpoint.pth
                phase2_dir = os.path.dirname(phase2_ckpt_path)
                phase2_teacher_ckpt_path = os.path.join(
                    f"{phase2_dir}_teacher", "checkpoint.pth"
                )

            self._load_checkpoint_to_model(
                self.teacher_model, phase2_teacher_ckpt_path, strict=False
            )
            self._freeze_teacher()
            return self._train_with_distillation(setting)

        # Optional fallback: run full two-stage if explicitly requested.
        return self._train_two_stage(setting)

    # ------------------------------------------------------------------
    # Phase 1: standard supervised training
    # ------------------------------------------------------------------
    def _train_standard(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            epoch_time = time.time()

            for i, (batch_X, batch_P, batch_rul, batch_meta) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        rul_pred = self.model(batch_X)
                        loss = criterion(rul_pred.squeeze(-1), batch_rul)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    rul_pred = self.model(batch_X)
                    loss = criterion(rul_pred.squeeze(-1), batch_rul)
                    loss.backward()
                    optimizer.step()

                train_losses.append(loss.item())

                if (i + 1) % 50 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

            train_loss = np.average(train_losses)
            vali_loss = self._validate(vali_loader, criterion)
            print(f"Epoch: {epoch+1} cost: {time.time()-epoch_time:.1f}s | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    def _validate(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_X, batch_P, batch_rul, batch_meta in vali_loader:
                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device)

                rul_pred = self.model(batch_X)
                loss = criterion(rul_pred.squeeze(-1), batch_rul)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    # ------------------------------------------------------------------
    # Phase 2/3: two-stage distillation
    # ------------------------------------------------------------------
    def _train_two_stage(self, setting):
        print("=== Two-stage distillation training ===")

        # Stage 1: train teacher
        print("=== Stage 1: Train teacher model ===")
        teacher_setting = setting + "_teacher"
        self._train_teacher(teacher_setting)

        # Stage 2: freeze teacher, train student with distillation
        print("=== Stage 2: Knowledge distillation ===")
        self._freeze_teacher()
        return self._train_with_distillation(setting)

    # ---- Stage 1: teacher training ----
    def _train_teacher(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # Enable teacher params
        for param in self.teacher_model.parameters():
            param.requires_grad = True

        teacher_optim = optim.Adam(self.teacher_model.parameters(),
                                   lr=self.args.learning_rate)
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        teacher_epochs = getattr(self.args, 'teacher_pretrain_epochs',
                                 self.args.train_epochs // 2)

        for epoch in range(teacher_epochs):
            self.teacher_model.train()
            train_losses = []
            epoch_time = time.time()

            for i, (batch_X, batch_P, batch_rul, batch_meta) in enumerate(train_loader):
                teacher_optim.zero_grad()
                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        teacher_pred, _ = self.teacher_model(batch_X)
                        loss = criterion(teacher_pred.squeeze(-1), batch_rul)
                    scaler.scale(loss).backward()
                    scaler.step(teacher_optim)
                    scaler.update()
                else:
                    teacher_pred, _ = self.teacher_model(batch_X)
                    loss = criterion(teacher_pred.squeeze(-1), batch_rul)
                    loss.backward()
                    teacher_optim.step()

                train_losses.append(loss.item())

                if (i + 1) % 50 == 0:
                    print(f"\tTeacher iters: {i+1}, epoch: {epoch+1} | "
                          f"loss: {loss.item():.7f}")

            train_loss = np.average(train_losses)
            vali_loss = self._validate_teacher(vali_loader, criterion)
            print(f"Teacher Epoch: {epoch+1} cost: {time.time()-epoch_time:.1f}s | "
                  f"Train: {train_loss:.7f} Vali: {vali_loss:.7f}")

            early_stopping(vali_loss, self.teacher_model, path)
            if early_stopping.early_stop:
                print("Teacher early stopping")
                break
            adjust_learning_rate(teacher_optim, epoch + 1, self.args)

        best_path = os.path.join(path, 'checkpoint.pth')
        self.teacher_model.load_state_dict(torch.load(best_path))
        print("Teacher training complete, best model loaded.")

    def _validate_teacher(self, vali_loader, criterion):
        self.teacher_model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_X, batch_P, batch_rul, batch_meta in vali_loader:
                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device)

                teacher_pred, _ = self.teacher_model(batch_X)
                loss = criterion(teacher_pred.squeeze(-1), batch_rul)
                total_loss.append(loss.item())
        self.teacher_model.train()
        return np.average(total_loss)

    # ---- Stage 2: student distillation training ----
    def _train_with_distillation(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizers = self._select_optimizer()
        if isinstance(optimizers, tuple):
            model_optim, distill_optim = optimizers
        else:
            model_optim, distill_optim = optimizers, None

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            loss_tracking = {k: [] for k in
                            ['feature_loss', 'fcst_loss', 'recon_loss', 'att_loss',
                             'deg_weight_mean']}
            epoch_time = time.time()
            epoch_log_path = os.path.join(path, 'train_epoch_log.csv')

            for i, (batch_X, batch_P, batch_rul, batch_meta) in enumerate(train_loader):
                model_optim.zero_grad()
                if distill_optim:
                    distill_optim.zero_grad()

                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device).unsqueeze(-1)  # [B, 1]
                batch_P = batch_P.float().to(self.device) if self.phase >= 3 else None

                # --- Teacher forward (frozen, no grad, always K=1) ---
                # When K-window: teacher uses only the last image
                teacher_X = batch_X[:, -1] if batch_X.dim() == 5 else batch_X
                with torch.no_grad():
                    if self.phase >= 3:
                        teacher_pred, teacher_feat, teacher_patches = \
                            self.teacher_model(teacher_X, return_patch_tokens=True)
                    else:
                        teacher_pred, teacher_feat = self.teacher_model(teacher_X)
                        teacher_patches = None

                # --- Student forward with features (handles both K=1 and K>1) ---
                student_pred, student_feat, student_attn = \
                    self.model.forward_with_features(
                        batch_X, P=batch_P,
                        teacher_vision_feat=teacher_feat,
                        teacher_patch_tokens=teacher_patches,
                        return_attention=True,
                    )

                # --- Build teacher attention target for Phase 3 ---
                if self.phase >= 3 and student_attn is not None:
                    teacher_attn = self._build_teacher_attention_target(
                        student_attn, teacher_patches
                    )
                    if teacher_attn is None:
                        # Fallback: keep a valid target when patch tokens are unavailable.
                        uniform_prob = 1.0 / student_attn.size(-1)
                        teacher_attn = torch.full_like(student_attn, uniform_prob)
                else:
                    teacher_attn = None

                # --- Distillation loss (reuse existing framework) ---
                total_loss, loss_items = self.distill_loss.compute_total_loss(
                    ts_enc=student_feat,        # [B, d_model]
                    prompt_enc=teacher_feat,     # [B, 768]
                    ts_out=student_pred,         # [B, 1]
                    prompt_out=teacher_pred,     # [B, 1]
                    ts_att=student_attn,
                    prompt_att=teacher_attn,     # same shape as student_attn or None
                    real=batch_rul,              # [B, 1]
                    batch_meta=batch_meta,
                )

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    if distill_optim:
                        scaler.step(distill_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()
                    if distill_optim:
                        distill_optim.step()

                train_losses.append(total_loss.item())
                for k in loss_tracking:
                    if k in loss_items:
                        loss_tracking[k].append(loss_items[k])

                if (i + 1) % 50 == 0:
                    parts = [f"iters: {i+1}, epoch: {epoch+1} | "
                             f"loss: {total_loss.item():.7f}"]
                    for k in ['feature_loss', 'fcst_loss', 'recon_loss', 'att_loss',
                              'deg_weight_mean']:
                        if k in loss_items:
                            parts.append(f"{k}: {loss_items[k]:.7f}")
                    if hasattr(self.distill_loss, 'get_current_weights'):
                        w = self.distill_loss.get_current_weights()
                        parts.append(f"T={w['temperature']:.2f}")
                    print("\t" + ", ".join(parts))

            train_loss = np.average(train_losses)

            # Log per-component losses
            log = f"Epoch {epoch+1} | Train: {train_loss:.7f}"
            for k, v in loss_tracking.items():
                if v:
                    log += f", {k}: {np.average(v):.7f}"
            print(log)

            if (epoch + 1) % 5 == 0 and hasattr(self.distill_loss, 'print_loss_statistics'):
                self.distill_loss.print_loss_statistics()

            vali_loss = self._validate_with_distillation(vali_loader)
            print(f"Epoch: {epoch+1} | Train: {train_loss:.7f} Vali: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if distill_optim:
                adjust_learning_rate(distill_optim, epoch + 1, self.args)

            current_weights = (self.distill_loss.get_current_weights()
                               if hasattr(self.distill_loss, 'get_current_weights')
                               else {})
            fusion_alpha = self._get_phase3_fusion_alpha()
            self._append_epoch_log(
                epoch_log_path,
                {
                    'epoch': epoch + 1,
                    'model_lr': model_optim.param_groups[0]['lr'],
                    'distill_lr': (distill_optim.param_groups[0]['lr']
                                   if distill_optim else ''),
                    'train_total': train_loss,
                    'val_total': vali_loss,
                    'feature_loss': np.average(loss_tracking['feature_loss'])
                    if loss_tracking['feature_loss'] else '',
                    'fcst_loss': np.average(loss_tracking['fcst_loss'])
                    if loss_tracking['fcst_loss'] else '',
                    'recon_loss': np.average(loss_tracking['recon_loss'])
                    if loss_tracking['recon_loss'] else '',
                    'att_loss': np.average(loss_tracking['att_loss'])
                    if loss_tracking['att_loss'] else '',
                    'deg_weight_mean': np.average(loss_tracking['deg_weight_mean'])
                    if loss_tracking['deg_weight_mean'] else '',
                    'feature_w': current_weights.get('feature_w', ''),
                    'fcst_w': current_weights.get('fcst_w', ''),
                    'recon_w': current_weights.get('recon_w', ''),
                    'att_w': current_weights.get('att_w', ''),
                    'temperature': current_weights.get('temperature', ''),
                    'phase3_fusion_alpha': fusion_alpha if fusion_alpha is not None else '',
                }
            )

        best_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_path))
        return self.model

    def _validate_with_distillation(self, vali_loader):
        """Validate using distillation loss for consistency with training."""
        self.model.eval()
        self.teacher_model.eval()
        total_loss = []

        with torch.no_grad():
            for batch_X, batch_P, batch_rul, batch_meta in vali_loader:
                batch_X = batch_X.float().to(self.device)
                batch_rul = batch_rul.float().to(self.device).unsqueeze(-1)
                batch_P = batch_P.float().to(self.device) if self.phase >= 3 else None

                # Teacher always K=1
                teacher_X = batch_X[:, -1] if batch_X.dim() == 5 else batch_X
                if self.phase >= 3:
                    teacher_pred, teacher_feat, teacher_patches = \
                        self.teacher_model(teacher_X, return_patch_tokens=True)
                else:
                    teacher_pred, teacher_feat = self.teacher_model(teacher_X)
                    teacher_patches = None

                student_pred, student_feat, student_attn = \
                    self.model.forward_with_features(
                        batch_X, P=batch_P,
                        teacher_vision_feat=teacher_feat,
                        teacher_patch_tokens=teacher_patches,
                        return_attention=(self.phase >= 3),
                    )

                # Teacher attention target for Phase 3
                teacher_attn = None
                if self.phase >= 3 and student_attn is not None:
                    teacher_attn = self._build_teacher_attention_target(
                        student_attn, teacher_patches
                    )
                    if teacher_attn is None:
                        uniform_prob = 1.0 / student_attn.size(-1)
                        teacher_attn = torch.full_like(student_attn, uniform_prob)

                loss, _ = self.distill_loss.compute_total_loss(
                    ts_enc=student_feat,
                    prompt_enc=teacher_feat,
                    ts_out=student_pred,
                    prompt_out=teacher_pred,
                    ts_att=student_attn,
                    prompt_att=teacher_attn,
                    real=batch_rul,
                    batch_meta=batch_meta,
                )
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data('test')

        if test:
            ckpt_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device))

        self.model.eval()
        preds, trues = [], []
        all_bearing_ids, all_timesteps = [], []

        with torch.no_grad():
            for batch_X, batch_P, batch_rul, batch_meta in test_loader:
                batch_X = batch_X.float().to(self.device)
                batch_P = batch_P.float().to(self.device) if self.phase >= 3 else None

                rul_pred = self.model(batch_X, batch_P)
                preds.append(rul_pred.cpu().numpy())
                trues.append(batch_rul.numpy())
                all_bearing_ids.extend(batch_meta['bearing_id'])
                all_timesteps.extend(batch_meta['timestep'].cpu().numpy().tolist())

        preds = np.concatenate(preds, axis=0).flatten()
        trues = np.concatenate(trues, axis=0).flatten()

        # Metrics
        mse = np.mean((preds - trues) ** 2)
        mae = np.mean(np.abs(preds - trues))
        rmse = np.sqrt(mse)

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        eps_smape = 0.01
        smape = np.mean(2 * np.abs(preds - trues) /
                        (np.abs(preds) + np.abs(trues) + eps_smape)) * 100

        # NASA-style asymmetric scoring
        diff = preds - trues
        score = np.sum(np.where(diff < 0,
                                np.exp(-diff / 13) - 1,
                                np.exp(diff / 10) - 1))

        print(f"Test Results | MSE: {mse:.6f}, MAE: {mae:.6f}, "
              f"RMSE: {rmse:.6f}, SMAPE: {smape:.2f}%, Score: {score:.2f}")

        # Save
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'metrics.npy',
                np.array([mse, mae, rmse, score, smape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'test_timesteps.npy', np.array(all_timesteps, dtype=np.int32))
        np.save(folder_path + 'test_bearing_ids.npy', np.array(all_bearing_ids, dtype=object))

        # Append to results file
        with open("result_rul.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"MSE: {mse:.6f}, MAE: {mae:.6f}, "
                    f"RMSE: {rmse:.6f}, SMAPE: {smape:.2f}%, Score: {score:.2f}\n\n")

        # Save detailed metrics to txt file
        profile_data = self._save_model_profile(setting, folder_path)

        metrics_txt_path = os.path.join(folder_path, 'metrics.txt')
        with open(metrics_txt_path, 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Primary metrics:\n")
            f.write(f"  RMSE:  {rmse:.6f}\n")
            f.write(f"  SMAPE: {smape:.2f}%\n")
            f.write(f"{'='*50}\n")
            f.write(f"Secondary metrics:\n")
            f.write(f"  MSE:   {mse:.6f}\n")
            f.write(f"  MAE:   {mae:.6f}\n")
            f.write(f"  Score: {score:.2f}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Num samples: {len(preds)}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Model Profile:\n")
            if 'teacher' in profile_data and profile_data['teacher']:
                t = profile_data['teacher']
                f.write(format_profile_line(
                    'Teacher', t.get('vlm_type', '?'), t.get('hidden_size', '?'), t) + '\n')
            if 'student' in profile_data and profile_data['student']:
                s = profile_data['student']
                f.write(format_profile_line(
                    'Student', s.get('vision_type', '?'), s.get('hidden_size', '?'), s) + '\n')
            f.write(f"  K={profile_data.get('k_window_size', 1)}, "
                    f"L={profile_data.get('temporal_conv_layers', 2)}\n")
        print(f"Metrics saved to {metrics_txt_path}")

        return
