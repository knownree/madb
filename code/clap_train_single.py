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
EMB_DIR = "clap/emb/"
SAVE_DIR = "clap/train/se/"
SHARED_INIT_PT = ""

ID_COL = "id"
PEARSON_LAMBDA = 0.5
TARGET = "avg_sound_effect_perception"

DEVICE = "cuda"
SEED = 120
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-4
VALID_RATIO = 0.2
TOP_K = 5

MODEL_KWARGS = dict(
    input_dim=512,
    hidden_dims=64,
    mlp_hidden_dims=[64, 16],
    seq_len=8,
    nhead=8,
    num_layers=4,
    dim_feedforward=256,
    dropout=0.2,
    out_dim=1,
)

logging.basicConfig(level=logging.INFO)


# =========================================================
# ========================== UTIL ==========================
# =========================================================

def pearson_corr_loss(pred, target, eps=1e-8):
    """
    return 1 - Pearson
    """

    pred = pred.view(-1)
    target = target.view(-1)

    pred_mean = torch.mean(pred)
    target_mean = torch.mean(target)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = torch.sum(pred_centered * target_centered)

    denom = torch.sqrt(
        torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + eps
    )

    corr = numerator / (denom + eps)

    return 1.0 - corr


def save_config():
    cfg = {
        "CSV_PATH": CSV_PATH,
        "EMB_DIR": EMB_DIR,
        "SAVE_DIR": SAVE_DIR,
        "SHARED_INIT_PT": SHARED_INIT_PT,

        "ID_COL": ID_COL,
        "TARGET": TARGET,
        "loss": {
            "type": "mse + lambda * (1 - pearson)",
            "pearson_lambda": PEARSON_LAMBDA
        },
        "DEVICE": DEVICE,
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "VALID_RATIO": VALID_RATIO,
        "TOP_K": TOP_K,

        "MODEL_KWARGS": MODEL_KWARGS,

        "data_filtering": {
            "zero_removed": True,
            "description": "samples with TARGET == 0 are removed before training"
        },

        "metric": {
            "mse": True,
            "pearson": True,
            "spearman": True,
            "kendall": True,
            "composite_formula": "mse + 0.33*(1-pearson) + 0.33*(1-spearman) + 0.33*(1-kendall)"
        }
    }

    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_pt_path(i):
    return os.path.join(EMB_DIR, f"{int(float(i))}.pt")

def load_embedding(p):
    x = torch.load(p, map_location="cpu")
    if isinstance(x, dict):
        x = list(x.values())[0]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x.float()

def compute_metrics(pred, gt):
    mse = np.mean((pred - gt)**2)
    pearson = np.corrcoef(pred, gt)[0,1]
    spearman = spearmanr(gt, pred).correlation
    kendall = kendalltau(gt, pred).correlation

    composite = mse + 0.33*(1-pearson) + 0.33*(1-spearman) + 0.33*(1-kendall)

    return dict(
        mse=mse,
        pearson=pearson,
        spearman=spearman,
        kendall=kendall,
        composite=composite
    )


# =========================================================
# ========================= DATASET ========================
# =========================================================

class DS(Dataset):
    def __init__(self, df):
        self.data = []

        for _, r in df.iterrows():
            # save non-zero
            if float(r[TARGET]) == 0:
                continue

            p = build_pt_path(r[ID_COL])
            if not os.path.exists(p):
                continue

            y = float(r[TARGET])
            self.data.append((p, y))

        logging.info(f"{TARGET} usable samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        p, y = self.data[i]
        x = load_embedding(p)
        return x, torch.tensor(y, dtype=torch.float32)


def collate(b):
    xs, ys = zip(*b)
    return torch.stack(xs), torch.stack(ys)


# =========================================================
# =========================== TRAIN ========================
# =========================================================

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_config()
    df = pd.read_csv(CSV_PATH)

    # ===== split =====
    idx = np.random.permutation(len(df))
    n_val = int(len(df)*VALID_RATIO)
    train_df = df.iloc[idx[n_val:]]
    val_df = df.iloc[idx[:n_val]]

    train_loader = DataLoader(DS(train_df), BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(DS(val_df), BATCH_SIZE, shuffle=False, collate_fn=collate)

    # ===== model =====
    model = TransformerRegMLP5(**MODEL_KWARGS).to(DEVICE)

    if os.path.exists(SHARED_INIT_PT):
        state = torch.load(SHARED_INIT_PT, map_location="cpu")
        model.load_state_dict(state, strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    topk = []

    for epoch in range(EPOCHS):

        # ================= train =================
        model.train()
        for x, y in tqdm(train_loader, desc=f"train {epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            pred = model(x).view(-1)
            mse_loss = torch.mean((pred - y)**2)

            pearson_loss = pearson_corr_loss(pred, y)

            loss = mse_loss + PEARSON_LAMBDA * pearson_loss
            loss.backward()
            opt.step()

        # ================= val =================
        model.eval()

        preds, gts = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                p = model(x).view(-1).cpu().numpy()

                preds.append(p)
                gts.append(y.numpy())

        P = np.concatenate(preds)
        Y = np.concatenate(gts)

        m = compute_metrics(P, Y)

        logging.info(f"epoch {epoch} {m}")

        ep_dir = os.path.join(SAVE_DIR, f"epoch_{epoch}")
        os.makedirs(ep_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(ep_dir, "model.pt"))

        pd.DataFrame([m]).to_csv(os.path.join(ep_dir, "summary.csv"), index=False)

        topk.append((m["composite"], ep_dir))
        topk = sorted(topk, key=lambda x: x[0])

        while len(topk) > TOP_K:
            _, rm = topk.pop(-1)
            shutil.rmtree(rm, ignore_errors=True)

    rows = []
    for rank, (score, path) in enumerate(topk, 1):
        df = pd.read_csv(os.path.join(path, "summary.csv"))
        df["rank"] = rank
        rows.append(df)

    pd.concat(rows).to_csv(os.path.join(SAVE_DIR, "top5_summary.csv"), index=False)

    logging.info("DONE")


if __name__ == "__main__":
    print(TARGET)
    main()