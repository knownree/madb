#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from transformer import TransformerRegMLP5


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_id(sample_id):
    sid = str(sample_id).strip()
    try:
        ff = float(sid)
        if ff.is_integer():
            sid = str(int(ff))
    except Exception:
        pass
    return sid


def build_pt_path(emb_dir, sample_id):
    return os.path.join(emb_dir, f"{sanitize_id(sample_id)}.pt")


def build_ckpt_path(save_dir, dim_name):
    safe_dim = dim_name.replace("/", "_").replace(" ", "_")
    return os.path.join(save_dir, "per_dimension_best_models", f"{safe_dim}.pt")


def load_embedding(path, input_dim):
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
            raise RuntimeError(f"No tensor found in dict pt: {path}")
    elif isinstance(x, (list, tuple)):
        emb = None
        for v in x:
            if isinstance(v, torch.Tensor):
                emb = v
                break
        if emb is None:
            raise RuntimeError(f"No tensor found in list/tuple pt: {path}")
    else:
        raise RuntimeError(f"Unsupported pt format: {path}")

    emb = emb.detach().float().cpu()

    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    elif emb.dim() == 2:
        pass
    else:
        emb = emb.reshape(-1, emb.shape[-1])

    if emb.shape[-1] != input_dim:
        raise RuntimeError(
            f"Expected embedding dim={input_dim}, got {tuple(emb.shape)} in {path}"
        )

    if emb.shape[0] != 1:
        emb = emb[:1]

    if not torch.isfinite(emb).all():
        raise RuntimeError(f"NaN/Inf found in embedding: {path}")

    return emb.contiguous()


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

    return {
        "n_samples": int(n_valid),
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
    }


class EmbeddingCSVDataset(Dataset):
    def __init__(self, df, csv_path, emb_dir, id_col, target_columns, input_dim):
        self.items = []
        self.emb_dir = emb_dir
        self.id_col = id_col
        self.target_columns = target_columns
        self.input_dim = input_dim

        missing = 0
        bad_label = 0

        for _, row in df.iterrows():
            pt_path = build_pt_path(emb_dir, row[id_col])

            if not os.path.exists(pt_path):
                missing += 1
                continue

            try:
                y = [float(row[c]) for c in target_columns]
            except Exception:
                bad_label += 1
                continue

            if not np.isfinite(y).all():
                bad_label += 1
                continue

            self.items.append(
                (
                    pt_path,
                    np.array(y, dtype=np.float32),
                    sanitize_id(row[id_col]),
                )
            )

        logging.info(
            f"Dataset from {csv_path}: valid={len(self.items)}, "
            f"missing_pt={missing}, bad_label={bad_label}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pt_path, y, sid = self.items[idx]
        x = load_embedding(pt_path, self.input_dim)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y, sid


def collate_fn(batch):
    xs, ys, sids = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    return xs, ys, list(sids)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def override_config(cfg, args):
    if args.csv_path is not None:
        cfg["CSV_PATH"] = args.csv_path
    if args.emb_dir is not None:
        cfg["EMB_DIR"] = args.emb_dir
    if args.save_dir is not None:
        cfg["SAVE_DIR"] = args.save_dir
    if args.seed is not None:
        cfg["SEED"] = args.seed
    if args.batch_size is not None:
        cfg["BATCH_SIZE"] = args.batch_size
    if args.num_workers is not None:
        cfg["NUM_WORKERS"] = args.num_workers
    if args.valid_ratio is not None:
        cfg["VALID_RATIO"] = args.valid_ratio
    if args.device is not None:
        cfg["DEVICE"] = args.device
    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=None)

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["val", "train", "all"],
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save metric csv.",
    )

    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Path to save gt, raw_pred, pred_score csv.",
    )

    args = parser.parse_args()

    cfg = load_json(args.config)
    cfg = override_config(cfg, args)

    csv_path = cfg["CSV_PATH"]
    emb_dir = cfg["EMB_DIR"]
    save_dir = cfg["SAVE_DIR"]

    id_col = cfg.get("ID_COL", "id")
    target_columns = cfg["TARGET_COLUMNS"]
    model_kwargs = cfg["MODEL_KWARGS"]

    seed = int(cfg.get("SEED", 42))
    batch_size = int(cfg.get("BATCH_SIZE", 64))
    num_workers = int(cfg.get("NUM_WORKERS", 2))
    valid_ratio = float(cfg.get("VALID_RATIO", 0.2))
    input_dim = int(model_kwargs.get("input_dim", 512))

    device_str = cfg.get("DEVICE", "cuda")
    device = torch.device(
        device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu"
    )

    set_seed(seed)

    ckpt_dir = os.path.join(save_dir, "per_dimension_best_models")

    logging.info(f"Using device: {device}")
    logging.info(f"CSV_PATH: {csv_path}")
    logging.info(f"EMB_DIR: {emb_dir}")
    logging.info(f"SAVE_DIR: {save_dir}")
    logging.info(f"CKPT_DIR: {ckpt_dir}")
    logging.info(f"Split: {args.split}")
    logging.info(f"Seed: {seed}")

    df = pd.read_csv(csv_path)

    idx = np.random.permutation(len(df))
    n_val = max(1, int(len(df) * valid_ratio))

    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    train_df = df.iloc[idx[n_val:]].reset_index(drop=True)

    if args.split == "val":
        eval_df = val_df
    elif args.split == "train":
        eval_df = train_df
    else:
        eval_df = df.reset_index(drop=True)

    dataset = EmbeddingCSVDataset(
        df=eval_df,
        csv_path=csv_path,
        emb_dir=emb_dir,
        id_col=id_col,
        target_columns=target_columns,
        input_dim=input_dim,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    models = []

    for dim_name in target_columns:
        model = TransformerRegMLP5(**model_kwargs).to(device)

        pt_path = build_ckpt_path(save_dir, dim_name)

        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Cannot find checkpoint for {dim_name}: {pt_path}"
            )

        state = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()

        models.append(model)
        logging.info(f"Loaded [{dim_name}] from {pt_path}")

    all_gt = []
    all_raw_pred = [[] for _ in target_columns]
    all_ids = []

    with torch.no_grad():
        for x, y, sids in tqdm(loader, desc="evaluating"):
            x = x.to(device, non_blocking=True)

            all_gt.append(y.numpy())
            all_ids.extend(sids)

            for i, model in enumerate(models):
                raw_pred = model(x).view(-1).detach().cpu().numpy()
                all_raw_pred[i].append(raw_pred)

    Y = np.concatenate(all_gt, axis=0)
    RAW_P = [np.concatenate(p, axis=0) for p in all_raw_pred]

    rows = []
    pred_df = pd.DataFrame({id_col: all_ids})

    for i, dim_name in enumerate(target_columns):
        gt = Y[:, i]
        raw_pred = RAW_P[i]

        valid_mask = gt != 0

        pred_score = raw_pred.copy()
        pred_score[~valid_mask] = 0.0

        metrics = compute_metrics(
            pred_score[valid_mask],
            gt[valid_mask],
        )

        rows.append({
            "dimension": dim_name,
            "n_total": int(len(gt)),
            "n_valid": int(valid_mask.sum()),
            "n_zero_gt": int((~valid_mask).sum()),
            "mse": metrics["mse"],
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "kendall": metrics["kendall"],
        })

        pred_df[f"gt_{dim_name}"] = gt
        pred_df[f"raw_pred_{dim_name}"] = raw_pred
        pred_df[f"pred_score_{dim_name}"] = pred_score
        pred_df[f"valid_mask_{dim_name}"] = valid_mask.astype(int)

    result_df = pd.DataFrame(rows)

    mean_row = {
        "dimension": "MEAN",
        "n_total": int(np.nanmean(result_df["n_total"])),
        "n_valid": int(np.nanmean(result_df["n_valid"])),
        "n_zero_gt": int(np.nanmean(result_df["n_zero_gt"])),
        "mse": float(np.nanmean(result_df["mse"])),
        "pearson": float(np.nanmean(result_df["pearson"])),
        "spearman": float(np.nanmean(result_df["spearman"])),
        "kendall": float(np.nanmean(result_df["kendall"])),
    }

    result_df = pd.concat(
        [result_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )

    print("\n===== Evaluation Results =====")
    print(result_df.to_string(index=False))

    if args.output_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        result_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        logging.info(f"Saved metrics to: {args.output_csv}")

    if args.save_predictions is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_predictions)), exist_ok=True)
        pred_df.to_csv(args.save_predictions, index=False, encoding="utf-8-sig")
        logging.info(f"Saved predictions to: {args.save_predictions}")


if __name__ == "__main__":
    main()