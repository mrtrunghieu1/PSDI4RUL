"""
Experiment class for SOTA time-series RUL baselines.

Shared training/validation/test loop for PatchTST, TimesNet, TimeMixer, TimeMixerPP.
Inherits from Exp_Basic; uses importlib to load models so no static imports are needed.

Results are written in the same format as exp_rul.py:
  - checkpoints/{setting}/checkpoint.pth
  - results/{setting}/pred.npy, true.npy, metrics.npy
  - result_rul.txt  (appended)
"""

import importlib
import os
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Registry: model name → module path
MODEL_REGISTRY = {
    "PatchTST":    "models.ts_backbone.patch_tst",
    "TimesNet":    "models.ts_backbone.times_net",
    "TimeMixer":   "models.ts_backbone.time_mixer",
    "TimeMixerPP": "models.ts_backbone.time_mixer_pp",
}


class Exp_TS_RUL(Exp_Basic):
    """Experiment class for time-series sequence-to-one RUL regression."""

    def __init__(self, args):
        super().__init__(args)

    # ── Model construction ────────────────────────────────────────────────────

    def _build_model(self):
        model_name = self.args.model
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown TS model '{model_name}'. "
                f"Choose from: {list(MODEL_REGISTRY.keys())}"
            )
        module = importlib.import_module(MODEL_REGISTRY[model_name])
        model = module.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # ── Data ──────────────────────────────────────────────────────────────────

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    # ── Training utilities ────────────────────────────────────────────────────

    def _select_criterion(self):
        loss_name = getattr(self.args, "loss", "SmoothL1")
        if loss_name == "MSE":
            return nn.MSELoss()
        if loss_name == "MAE":
            return nn.L1Loss()
        return nn.SmoothL1Loss()

    def _select_optimizer(self):
        weight_decay = getattr(self.args, "weight_decay", 1e-5)
        return optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=weight_decay,
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for x, rul, _ in vali_loader:
                x = x.float().to(self.device)
                rul = rul.float().to(self.device)
                pred = self.model(x).squeeze(-1)
                total_loss.append(criterion(pred, rul).item())
        self.model.train()
        return float(np.mean(total_loss))

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")

        ckpt_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(ckpt_path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stop = EarlyStopping(patience=self.args.patience, verbose=True)

        use_amp = getattr(self.args, "use_amp", False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_losses = []
            t0 = time.time()

            for i, (x, rul, _) in enumerate(train_loader):
                optimizer.zero_grad()
                x = x.float().to(self.device)      # [B, C, L]
                rul = rul.float().to(self.device)  # [B]

                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x).squeeze(-1)
                        loss = criterion(pred, rul)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self.model(x).squeeze(-1)
                    loss = criterion(pred, rul)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                epoch_losses.append(loss.item())
                if (i + 1) % 50 == 0:
                    print(f"\titers: {i+1}/{len(train_loader)}, "
                          f"epoch: {epoch+1} | loss: {loss.item():.6f}")

            train_loss = float(np.mean(epoch_losses))
            vali_loss = self._validate(vali_loader, criterion)
            elapsed = time.time() - t0

            print(f"Epoch {epoch+1:3d}/{self.args.train_epochs} | "
                  f"cost: {elapsed:.1f}s | "
                  f"Train: {train_loss:.6f}  Val: {vali_loss:.6f}")

            early_stop(vali_loss, self.model, ckpt_path)
            if early_stop.early_stop:
                print("Early stopping triggered.")
                break

            adjust_learning_rate(optimizer, epoch + 1, self.args)

        # Restore best checkpoint
        best_ckpt = os.path.join(ckpt_path, "checkpoint.pth")
        self.model.load_state_dict(
            torch.load(best_ckpt, map_location=self.device)
        )
        return self.model

    # ── Test / Evaluation ─────────────────────────────────────────────────────

    def test(self, setting, test: int = 0):
        _, test_loader = self._get_data("test")

        if test:
            ckpt = os.path.join("./checkpoints", setting, "checkpoint.pth")
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device)
            )

        self.model.eval()
        preds_list, trues_list = [], []

        with torch.no_grad():
            for x, rul, _ in test_loader:
                x = x.float().to(self.device)
                pred = self.model(x).squeeze(-1).cpu().numpy()
                preds_list.append(pred)
                trues_list.append(rul.numpy())

        preds = np.concatenate(preds_list).flatten()
        trues = np.concatenate(trues_list).flatten()

        # ── Metrics ───────────────────────────────────────────────────────────
        mse = float(np.mean((preds - trues) ** 2))
        mae = float(np.mean(np.abs(preds - trues)))
        rmse = float(np.sqrt(mse))
        eps = 0.01
        smape = float(
            np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + eps))
            * 100
        )
        diff = preds - trues
        score = float(np.sum(np.where(diff < 0,
                                      np.exp(-diff / 13) - 1,
                                      np.exp(diff / 10) - 1)))

        # ── Save results ──────────────────────────────────────────────────────
        folder = os.path.join("./results", setting)
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "metrics.npy"),
                np.array([mse, mae, rmse, score, smape]))
        np.save(os.path.join(folder, "pred.npy"), preds)
        np.save(os.path.join(folder, "true.npy"), trues)

        # Append to summary file (same format as exp_rul.py)
        with open("result_rul.txt", "a") as f:
            f.write(f"{setting}\n")
            f.write(
                f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, "
                f"SMAPE: {smape:.2f}%, Score: {score:.2f}\n\n"
            )

        print(f"[{setting}]  "
              f"MSE:{mse:.6f}  MAE:{mae:.6f}  RMSE:{rmse:.6f}  "
              f"SMAPE:{smape:.2f}%  Score:{score:.2f}")

        # ── Visualisation ─────────────────────────────────────────────────────
        self._plot_results(trues, preds, folder, setting)

        return mse, mae, rmse, smape, score

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _plot_results(self, trues, preds, folder, setting):
        """Save prediction vs ground truth line plot and scatter plot."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Line plot
            axes[0].plot(trues, label="Ground Truth", linewidth=1.2, color="steelblue")
            axes[0].plot(preds, label="Prediction", linewidth=1.2,
                         color="tomato", alpha=0.8)
            axes[0].set_title(f"RUL Prediction — {self.args.model}")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("RUL")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Scatter plot
            axes[1].scatter(trues, preds, alpha=0.4, s=10, color="steelblue")
            lims = [0, 1]
            axes[1].plot(lims, lims, "r--", linewidth=1.0, label="y=x")
            axes[1].set_xlim(lims)
            axes[1].set_ylim(lims)
            axes[1].set_title("Predicted vs True RUL")
            axes[1].set_xlabel("True RUL")
            axes[1].set_ylabel("Predicted RUL")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.suptitle(setting, fontsize=9)
            plt.tight_layout()
            fig_path = os.path.join(folder, "prediction_plot.png")
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved: {fig_path}")
        except Exception as e:
            print(f"  Warning: could not save plot ({e})")
