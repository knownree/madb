#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau

# ===== import model =====
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformer import TransformerRegMLP5


# =========================================================
# ========================= CONFIG =========================
# =========================================================

CSV_PATH = "data/annotation/song_avg_scores.csv"
EMB_DIR = "muq/emb/"
SAVE_DIR = "muq/train/7/"
SHARED_INIT_PT = ""

ID_COL = "id"

TARGET_COLUMNS = [
    "avg_overall_score",
    "avg_melody_perception",
    "avg_melody_emotion",
    "avg_rhythm_perception",
    "avg_structure_perception",
    "avg_performance_and_singing_mood",
    "avg_performance_skill",
]

DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 0.0
VALID_RATIO = 0.2
USE_AMP = False

# ===== 训练损失 =====
USE_PEARSON_LOSS = True
PEARSON_LOSS_WEIGHT = 0.5   # 可调，比如 0.1 / 0.3 / 0.5 / 1.0

# ===== per-dim best 仍按这个综合指标选最优 =====
COMPOSITE_W_MSE = 1.0
COMPOSITE_W_PEARSON = 0.33
COMPOSITE_W_SPEARMAN = 0.33
COMPOSITE_W_KENDALL = 0.33

MODEL_KWARGS = dict(
    input_dim=1024,
    hidden_dims=256,
    mlp_hidden_dims=[256, 64],
    seq_len=8,
    nhead=8,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    out_dim=1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# =========================================================
# ========================== UTIL ==========================
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_name(x: str) -> str:
    return x.replace("/", "_").replace("\\", "_").replace(" ", "_")

def build_pt_path(sample_id):
    sid = str(sample_id).strip()
    try:
        ff = float(sid)
        if ff.is_integer():
            sid = str(int(ff))
    except Exception:
        pass
    return os.path.join(EMB_DIR, f"{sid}.pt")

def extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        tensor_keys = [k for k, v in obj.items() if isinstance(v, torch.Tensor)]
        if len(tensor_keys) > 0:
            return obj
    return obj

def load_shared_init_state(pt_path):
    if not pt_path or (not os.path.exists(pt_path)):
        logging.warning(f"shared init pt not found, random init used: {pt_path}")
        return None
    obj = torch.load(pt_path, map_location="cpu")
    state = extract_state_dict(obj)
    if not isinstance(state, dict):
        logging.warning(f"invalid shared init pt, random init used: {pt_path}")
        return None
    logging.info(f"Loaded shared init state: {pt_path}")
    return state

def load_embedding(path):
    x = torch.load(path, map_location="cpu")

    if isinstance(x, torch.Tensor):
        emb = x
    elif isinstance(x, dict):
        emb = None
        for k in ("embedding", "emb", "feat", "features", "output", "feature"):
            if k in x and isinstance(x[k], torch.Tensor):
                emb = x[k]
                break
        if emb is None:
            for v in x.values():
                if isinstance(v, torch.Tensor):
                    emb = v
                    break
        if emb is None:
            raise RuntimeError(f"no tensor found in dict pt: {path}")
    elif isinstance(x, (list, tuple)):
        emb = None
        for v in x:
            if isinstance(v, torch.Tensor):
                emb = v
                break
        if emb is None:
            raise RuntimeError(f"no tensor found in list/tuple pt: {path}")
    else:
        raise RuntimeError(f"unsupported pt format: {path}")

    emb = emb.detach().float().cpu()

    if emb.dim() == 1:
        emb = emb.unsqueeze(0)  # (1,1024)
    elif emb.dim() == 2:
        pass
    else:
        emb = emb.reshape(-1, emb.shape[-1])

    if emb.shape[-1] != 1024:
        raise RuntimeError(f"expected embedding dim=1024, got {tuple(emb.shape)} in {path}")

    if emb.shape[0] != 1:
        emb = emb[:1]

    return emb.contiguous()

def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def pearson_corr_loss(pred, target, eps=1e-8):
    """
    返回 1 - Pearson，可微
    pred, target: shape (B,)
    """
    pred = pred.view(-1)
    target = target.view(-1)

    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + eps
    )

    corr = numerator / (denominator + eps)
    return 1.0 - corr

def safe_corrcoef(x, y):
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom <= 1e-12:
        return np.nan
    return float((xm * ym).sum() / denom)

def compute_composite(mse, pearson, spearman, kendall):
    p = 0.0 if np.isnan(pearson) else pearson
    s = 0.0 if np.isnan(spearman) else spearman
    k = 0.0 if np.isnan(kendall) else kendall
    return float(
        COMPOSITE_W_MSE * mse
        + COMPOSITE_W_PEARSON * (1.0 - p)
        + COMPOSITE_W_SPEARMAN * (1.0 - s)
        + COMPOSITE_W_KENDALL * (1.0 - k)
    )

def compute_metrics(pred, gt):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)

    n_valid = len(gt)
    if n_valid == 0:
        return {
            "n_samples": 0,
            "mse": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "kendall": np.nan,
            "composite": np.nan,
        }

    mse = float(np.mean((pred - gt) ** 2))
    pearson = safe_corrcoef(pred, gt)

    try:
        spearman = float(spearmanr(gt, pred).correlation) if n_valid >= 2 else np.nan
    except Exception:
        spearman = np.nan

    try:
        kendall = float(kendalltau(gt, pred).correlation) if n_valid >= 2 else np.nan
    except Exception:
        kendall = np.nan

    composite = compute_composite(mse, pearson, spearman, kendall)

    return {
        "n_samples": int(n_valid),
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "composite": composite,
    }

def save_runtime_config():
    cfg = {
        "CSV_PATH": CSV_PATH,
        "EMB_DIR": EMB_DIR,
        "SAVE_DIR": SAVE_DIR,
        "SHARED_INIT_PT": SHARED_INIT_PT,
        "TARGET_COLUMNS": TARGET_COLUMNS,
        "ID_COL": ID_COL,
        "DEVICE": DEVICE,
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "VALID_RATIO": VALID_RATIO,
        "USE_AMP": USE_AMP,
        "MODEL_KWARGS": MODEL_KWARGS,
        "loss": {
            "use_mask": False,
            "use_pearson_loss": USE_PEARSON_LOSS,
            "pearson_loss_weight": PEARSON_LOSS_WEIGHT,
            "train_formula": "mse + pearson_weight * (1 - pearson)" if USE_PEARSON_LOSS else "mse_only",
        },
        "selection_metric": {
            "type": "per_dimension_best_only",
            "composite_formula": {
                "mse": COMPOSITE_W_MSE,
                "1_minus_pearson": COMPOSITE_W_PEARSON,
                "1_minus_spearman": COMPOSITE_W_SPEARMAN,
                "1_minus_kendall": COMPOSITE_W_KENDALL,
            },
        },
    }
    with open(os.path.join(SAVE_DIR, "runtime_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def copy_file(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            pass


# =========================================================
# ========================= DATASET ========================
# =========================================================

class EmbeddingCSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.items = []

        for _, row in df.iterrows():
            pt_path = build_pt_path(row[ID_COL])
            if not os.path.exists(pt_path):
                continue

            try:
                y = [float(row[c]) for c in TARGET_COLUMNS]
            except Exception:
                continue

            self.items.append((pt_path, np.array(y, dtype=np.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pt_path, y = self.items[idx]
        x = load_embedding(pt_path)  # (1,1024)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)  # (B,1,1024)
    ys = torch.stack(ys, dim=0)  # (B,num_dims)
    return xs, ys


# =========================================================
# ====================== MODEL HELPERS =====================
# =========================================================

def build_models(device, shared_state=None):
    models = []
    optimizers = []

    for dim_name in TARGET_COLUMNS:
        model = TransformerRegMLP5(**MODEL_KWARGS).to(device)
        if shared_state is not None:
            missing, unexpected = model.load_state_dict(shared_state, strict=False)
            logging.info(
                f"[{dim_name}] shared init loaded | missing={len(missing)} unexpected={len(unexpected)}"
            )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        models.append(model)
        optimizers.append(optimizer)

    return models, optimizers


# =========================================================
# =========================== MAIN =========================
# =========================================================

def main():
    set_seed(SEED)
    ensure_dir(SAVE_DIR)
    save_runtime_config()

    df = pd.read_csv(CSV_PATH)

    # ===== split =====
    idx = np.random.permutation(len(df))
    n_val = max(1, int(len(df) * VALID_RATIO))
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)

    train_ds = EmbeddingCSVDataset(train_df)
    val_ds = EmbeddingCSVDataset(val_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    shared_state = load_shared_init_state(SHARED_INIT_PT)
    models, optimizers = build_models(device, shared_state)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 每个维度单独最优
    best_by_dim = {}
    tmp_dimbest_dir = os.path.join(SAVE_DIR, "_tmp_dim_best")
    ensure_dir(tmp_dimbest_dir)

    for dim_name in TARGET_COLUMNS:
        best_by_dim[dim_name] = {
            "epoch": None,
            "mse": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "kendall": np.nan,
            "composite": np.inf,
            "n_samples": 0,
            "tmp_model_path": None,
        }

    for epoch in range(EPOCHS):
        # ===================== train =====================
        for m in models:
            m.train()

        train_loss_sum = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
        train_batch_count = np.zeros(len(TARGET_COLUMNS), dtype=np.int64)

        pbar = tqdm(train_loader, desc=f"train {epoch}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)   # (B,1,1024)
            y = y.to(device, non_blocking=True)   # (B,num_dims)

            show_log = {}

            for i, m in enumerate(models):
                optimizers[i].zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    pred = m(x).view(-1)
                    mse = mse_loss(pred, y[:, i])

                    if USE_PEARSON_LOSS:
                        p_loss = pearson_corr_loss(pred, y[:, i])
                        loss = mse + PEARSON_LOSS_WEIGHT * p_loss
                    else:
                        p_loss = None
                        loss = mse

                scaler.scale(loss).backward()
                scaler.step(optimizers[i])
                scaler.update()

                train_loss_sum[i] += loss.item()
                train_batch_count[i] += 1

                if USE_PEARSON_LOSS:
                    show_log[f"l{i}"] = f"{loss.item():.4f}"
                    show_log[f"m{i}"] = f"{mse.item():.4f}"
                else:
                    show_log[f"l{i}"] = f"{loss.item():.4f}"

            pbar.set_postfix(show_log)

        train_loss_avg = np.array([
            train_loss_sum[i] / train_batch_count[i] if train_batch_count[i] > 0 else np.nan
            for i in range(len(TARGET_COLUMNS))
        ], dtype=np.float64)

        # ===================== val =====================
        for m in models:
            m.eval()

        all_pred = [[] for _ in models]
        all_gt = []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"valid {epoch}"):
                x = x.to(device, non_blocking=True)
                all_gt.append(y.numpy())

                for i, m in enumerate(models):
                    p = m(x).view(-1).cpu().numpy()
                    all_pred[i].append(p)

        Y = np.concatenate(all_gt, axis=0)  # (N,num_dims)
        P = [np.concatenate(p, axis=0) for p in all_pred]

        summary_rows = []

        for i, dim_name in enumerate(TARGET_COLUMNS):
            metrics = compute_metrics(P[i], Y[:, i])

            row = {
                "dimension": dim_name,
                "epoch": epoch,
                "train_loss": train_loss_avg[i],
                "n_samples": metrics["n_samples"],
                "mse": metrics["mse"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
                "kendall": metrics["kendall"],
                "composite": metrics["composite"],
            }
            summary_rows.append(row)

            # ===== 每个维度单独 best =====
            if metrics["composite"] < best_by_dim[dim_name]["composite"]:
                old_path = best_by_dim[dim_name]["tmp_model_path"]
                if old_path is not None and os.path.exists(old_path):
                    os.remove(old_path)

                tmp_model_path = os.path.join(tmp_dimbest_dir, f"{sanitize_name(dim_name)}__epoch{epoch}.pt")
                torch.save(models[i].state_dict(), tmp_model_path)

                best_by_dim[dim_name] = {
                    "epoch": epoch,
                    "mse": metrics["mse"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "kendall": metrics["kendall"],
                    "composite": metrics["composite"],
                    "n_samples": metrics["n_samples"],
                    "tmp_model_path": tmp_model_path,
                }

        mean_mse = float(np.nanmean([r["mse"] for r in summary_rows])) if len(summary_rows) > 0 else np.nan
        mean_pearson = float(np.nanmean([r["pearson"] for r in summary_rows])) if len(summary_rows) > 0 else np.nan
        mean_spearman = float(np.nanmean([r["spearman"] for r in summary_rows])) if len(summary_rows) > 0 else np.nan
        mean_kendall = float(np.nanmean([r["kendall"] for r in summary_rows])) if len(summary_rows) > 0 else np.nan

        logging.info(
            f"epoch={epoch} "
            f"mean_mse={mean_mse:.6f} "
            f"mean_pearson={mean_pearson:.6f} "
            f"mean_spearman={mean_spearman:.6f} "
            f"mean_kendall={mean_kendall:.6f}"
        )

    # ===================== 最终输出 per-dimension best =====================
    per_dim_root = os.path.join(SAVE_DIR, "per_dimension_best_models")
    ensure_dir(per_dim_root)

    per_dim_config = {
        "source": "per_dimension_best_only",
        "selection_metric": "composite",
        "loss": {
            "use_mask": False,
            "use_pearson_loss": USE_PEARSON_LOSS,
            "pearson_loss_weight": PEARSON_LOSS_WEIGHT,
            "train_formula": "mse + pearson_weight * (1 - pearson)" if USE_PEARSON_LOSS else "mse_only",
        },
        "dimensions": {}
    }

    per_dim_rows = []

    for dim_name in TARGET_COLUMNS:
        info = best_by_dim[dim_name]
        dst_pt = os.path.join(per_dim_root, f"{sanitize_name(dim_name)}.pt")
        copy_file(info["tmp_model_path"], dst_pt)

        per_dim_config["dimensions"][dim_name] = {
            "epoch": info["epoch"],
            "mse": info["mse"],
            "pearson": info["pearson"],
            "spearman": info["spearman"],
            "kendall": info["kendall"],
            "composite": info["composite"],
            "n_samples": info["n_samples"],
            "pt_path": dst_pt,
        }

        per_dim_rows.append({
            "dimension": dim_name,
            "epoch": info["epoch"],
            "mse": info["mse"],
            "pearson": info["pearson"],
            "spearman": info["spearman"],
            "kendall": info["kendall"],
            "composite": info["composite"],
            "n_samples": info["n_samples"],
            "pt_path": dst_pt,
        })

    with open(os.path.join(per_dim_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(per_dim_config, f, ensure_ascii=False, indent=2)

    pd.DataFrame(per_dim_rows).to_csv(
        os.path.join(SAVE_DIR, "per_dimension_best_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 清理临时目录
    remove_path(tmp_dimbest_dir)

    logging.info("DONE")
    logging.info(f"Per-dimension best saved to: {per_dim_root}")


if __name__ == "__main__":
    main()
