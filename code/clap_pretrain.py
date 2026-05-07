#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import json
import math
import random
from copy import deepcopy
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import ClapModel, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel




BASE_MODEL = "laion/clap-htsat-unfused"


USE_PURE_CLAP_CKPT = False
PURE_CLAP_CKPT_PATH = None 

CSV_PATH = "data/annotation/MADB_data_id_f_en7.csv"
AUDIO_DIR = "clap/wav/"
SAVE_DIR = "clap/pretrain/"

ID_COL = "id"
AUDIO_EXT = ".wav"

COMMENT_PREFIX = "comment_eng"
GENRE_PREFIX = "genre"
MOOD_PREFIX = "mood"

SAMPLE_RATE = 48000


SKIP_CUDA_OOM = True
MAX_OOM_WARNINGS_PER_EPOCH = 20


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 15

TRAIN_RATIO = 0.85
PRINT_FREQ = 50
SAVE_EVERY_EPOCH = True
MAX_GRAD_NORM = 1.0

LR_NEW_MODULES = 1e-4
LR_LORA = 3e-5
WEIGHT_DECAY = 1e-4


WARMUP_RATIO = 0.05


USE_TAGS = False
FINETUNE_TEXT_ENCODER = False
FINETUNE_AUDIO_ENCODER = True

LOSS_WEIGHT_AUDIO_FUSED = 1.0
LOSS_WEIGHT_AUDIO_TEXT = 0.3
LOSS_WEIGHT_AUDIO_TAG = 0.2


AUDIO_POOLING_MODE = "whole_audio_mean"

CHUNK_SECONDS = 10.0
DROP_LAST_CHUNK_IF_SHORT = False
PAD_SHORT_AUDIO_TO_CHUNK = True


WHOLE_AUDIO_MAX_SECONDS = None

USE_LORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

COMMON_EMB_DIM = 512
COMMENT_POOL_HIDDEN_DIM = 512
AUDIO_CHUNK_POOL_HIDDEN_DIM = 512

TAG_EMB_DIM = 16
TAG_PROJ_HIDDEN_DIM = 128

USE_JSON_VOCAB = False
GENRE_VOCAB_JSON = ""
MOOD_VOCAB_JSON = ""

GENRE_LABELS = [
    "rock/metal",
    "blues/jazz/souls",
    "pop",
    "chinese pop",
    "classical",
    "chinese classical",
    "dj",
    "hiphop/rap",
    "country",
    "electronic",
]

MOOD_LABELS = [
    "passionable",
    "peaceful",
    "happy",
    "sad",
    "angry",
    "nervous",
]

# ---------------- retrieval ----------------
RETRIEVAL_KS = [1, 5, 10]


PRINT_TRAINABLE_PARAMS = True



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_label(x: str) -> str:
    return str(x).strip().replace("，", ",").lower()


def load_vocab_from_list(labels: List[str]) -> Dict[str, int]:
    vocab = {}
    for i, x in enumerate(labels):
        key = normalize_label(x)
        if key and key not in vocab:
            vocab[key] = i
    return vocab

def is_cuda_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    oom_keywords = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    ]
    return any(k in msg for k in oom_keywords)


def cleanup_after_oom():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_vocab_maybe_json(use_json: bool, json_path: str, fallback_labels: List[str]) -> Dict[str, int]:
    if use_json and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            vocab = {}
            for i, x in enumerate(obj):
                key = normalize_label(x)
                if key and key not in vocab:
                    vocab[key] = i
            return vocab

        if isinstance(obj, dict):
            vocab = {}
            for k, v in obj.items():
                key = normalize_label(k)
                if key and key not in vocab:
                    vocab[key] = int(v)
            return vocab

        raise ValueError(f"Unsupported vocab format: {json_path}")

    return load_vocab_from_list(fallback_labels)


def print_trainable_parameters(model: nn.Module):
    total_params = 0
    trainable_params = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
    ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    print(f"[Params] trainable: {trainable_params:,}")
    print(f"[Params] total:     {total_params:,}")
    print(f"[Params] ratio:     {ratio:.4f}%")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out




def safe_to_list(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []

    s = s.replace("，", ",")

    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [normalize_label(x) for x in obj if str(x).strip()]
        except Exception:
            pass

    if "," in s:
        return [normalize_label(x) for x in s.split(",") if str(x).strip()]

    return [normalize_label(s)]


def collect_sequential_columns_from_row(row: pd.Series, prefix: str) -> List[str]:
    vals = []
    i = 1
    while True:
        col = f"{prefix}{i}"
        if col not in row.index:
            break
        cell_items = safe_to_list(row[col])
        for item in cell_items:
            item = item.strip()
            if item:
                vals.append(item)
        i += 1
    return vals


def collect_comment_list(row: pd.Series, prefix: str = "comment") -> List[str]:
    comments = collect_sequential_columns_from_row(row, prefix)
    comments = [x.strip() for x in comments if str(x).strip()]
    return comments


def extract_label_ids_from_row(
    row: pd.Series,
    genre_vocab: Dict[str, int],
    mood_vocab: Dict[str, int],
    genre_prefix: str = "genre",
    mood_prefix: str = "mood",
):
    raw_genres = collect_sequential_columns_from_row(row, genre_prefix)
    raw_moods = collect_sequential_columns_from_row(row, mood_prefix)

    genre_ids = [genre_vocab[g] for g in raw_genres if g in genre_vocab]
    mood_ids = [mood_vocab[m] for m in raw_moods if m in mood_vocab]

    return {
        "genre_ids": genre_ids,
        "mood_ids": mood_ids,
        "raw_genres": raw_genres,
        "raw_moods": raw_moods,
    }




def pad_or_truncate_to_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    if len(wav) == target_len:
        return wav
    if len(wav) > target_len:
        return wav[:target_len]
    pad_len = target_len - len(wav)
    return np.pad(wav, (0, pad_len), mode="constant")


def make_chunks_full_cover(
    wav: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    drop_last_if_short: bool = False,
    pad_short_audio_to_chunk: bool = True,
) -> List[np.ndarray]:
    chunk_len = int(sample_rate * chunk_seconds)
    total_len = len(wav)

    if total_len == 0:
        return [np.zeros(chunk_len, dtype=np.float32)]

    if total_len < chunk_len:
        if pad_short_audio_to_chunk:
            return [pad_or_truncate_to_length(wav, chunk_len).astype(np.float32)]
        return [wav.astype(np.float32)]

    chunks = []
    start = 0
    while start < total_len:
        end = start + chunk_len
        piece = wav[start:end]
        if len(piece) < chunk_len:
            if drop_last_if_short:
                break
            piece = pad_or_truncate_to_length(piece, chunk_len)
        chunks.append(piece.astype(np.float32))
        start += chunk_len

    if len(chunks) == 0:
        chunks = [pad_or_truncate_to_length(wav, chunk_len).astype(np.float32)]

    return chunks


def build_audio_units(
    wav: np.ndarray,
    sample_rate: int,
    mode: str,
    chunk_seconds: float,
    drop_last_if_short: bool,
    pad_short_audio_to_chunk: bool,
    whole_audio_max_seconds: Optional[float] = None,
) -> List[np.ndarray]:
    if mode in ["chunk_mean", "chunk_attention"]:
        return make_chunks_full_cover(
            wav=wav,
            sample_rate=sample_rate,
            chunk_seconds=chunk_seconds,
            drop_last_if_short=drop_last_if_short,
            pad_short_audio_to_chunk=pad_short_audio_to_chunk,
        )

    if mode == "whole_audio_mean":
        if whole_audio_max_seconds is not None:
            max_len = int(sample_rate * whole_audio_max_seconds)
            wav = wav[:max_len]
        if len(wav) == 0:
            wav = np.zeros(int(sample_rate * chunk_seconds), dtype=np.float32)
        return [wav.astype(np.float32)]

    raise ValueError(f"Unsupported AUDIO_POOLING_MODE: {mode}")


# =========================================================
# ===================== Dataset ========================
# =========================================================

class ClapTagDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        genre_vocab: Dict[str, int],
        mood_vocab: Dict[str, int],
        id_col: str = "id",
        audio_ext: str = ".wav",
        sample_rate: int = 48000,
        comment_prefix: str = "comment_eng",
        genre_prefix: str = "genre",
        mood_prefix: str = "mood",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.genre_vocab = genre_vocab
        self.mood_vocab = mood_vocab
        self.id_col = id_col
        self.audio_ext = audio_ext
        self.sample_rate = sample_rate
        self.comment_prefix = comment_prefix
        self.genre_prefix = genre_prefix
        self.mood_prefix = mood_prefix

        self.items = []
        missing_audio = 0
        empty_comment = 0

        for _, row in self.df.iterrows():
            audio_id = str(row[self.id_col]).strip()
            wav_path = os.path.join(self.audio_dir, audio_id + self.audio_ext)

            if not os.path.exists(wav_path):
                missing_audio += 1
                continue

            comments = collect_comment_list(row, self.comment_prefix)
            if len(comments) == 0:
                empty_comment += 1
                continue

            label_info = extract_label_ids_from_row(
                row=row,
                genre_vocab=self.genre_vocab,
                mood_vocab=self.mood_vocab,
                genre_prefix=self.genre_prefix,
                mood_prefix=self.mood_prefix,
            )

            self.items.append({
                "id": audio_id,
                "wav_path": wav_path,
                "comments": comments,
                "genre_ids": label_info["genre_ids"],
                "mood_ids": label_info["mood_ids"],
                "raw_genres": label_info["raw_genres"],
                "raw_moods": label_info["raw_moods"],
            })

        print(f"[Dataset] total valid samples: {len(self.items)}")
        print(f"[Dataset] empty comments:     {empty_comment}")

    def __len__(self):
        return len(self.items)

    def _load_audio(self, path: str) -> np.ndarray:
        wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return wav.astype(np.float32)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        wav = self._load_audio(item["wav_path"])
        return {
            "id": item["id"],
            "audio": wav,
            "comments": item["comments"],
            "genre_ids": item["genre_ids"],
            "mood_ids": item["mood_ids"],
            "raw_genres": item["raw_genres"],
            "raw_moods": item["raw_moods"],
        }


# =========================================================
# ===================== Collator =======================
# =========================================================

class ClapCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        sample_rate: int = 48000,
        audio_pooling_mode: str = "chunk_mean",
        chunk_seconds: float = 10.0,
        drop_last_if_short: bool = False,
        pad_short_audio_to_chunk: bool = True,
        whole_audio_max_seconds: Optional[float] = None,
    ):
        self.processor = processor
        self.sample_rate = sample_rate
        self.audio_pooling_mode = audio_pooling_mode
        self.chunk_seconds = chunk_seconds
        self.drop_last_if_short = drop_last_if_short
        self.pad_short_audio_to_chunk = pad_short_audio_to_chunk
        self.whole_audio_max_seconds = whole_audio_max_seconds

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [x["id"] for x in batch]

        comments_per_sample = [x["comments"] for x in batch]
        num_comments = [len(x) for x in comments_per_sample]

        flat_comments = []
        for comments in comments_per_sample:
            flat_comments.extend(comments)

        text_inputs = self.processor.tokenizer(
            flat_comments,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        genre_ids = [x["genre_ids"] for x in batch]
        mood_ids = [x["mood_ids"] for x in batch]
        raw_genres = [x["raw_genres"] for x in batch]
        raw_moods = [x["raw_moods"] for x in batch]

        audio_units_per_sample = []
        num_audio_units = []

        for x in batch:
            wav = x["audio"]
            units = build_audio_units(
                wav=wav,
                sample_rate=self.sample_rate,
                mode=self.audio_pooling_mode,
                chunk_seconds=self.chunk_seconds,
                drop_last_if_short=self.drop_last_if_short,
                pad_short_audio_to_chunk=self.pad_short_audio_to_chunk,
                whole_audio_max_seconds=self.whole_audio_max_seconds,
            )
            audio_units_per_sample.append(units)
            num_audio_units.append(len(units))

        flat_audio_units = []
        for units in audio_units_per_sample:
            flat_audio_units.extend(units)

        audio_inputs = self.processor.feature_extractor(
            flat_audio_units,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        out = {
            "ids": ids,
            "comments_per_sample": comments_per_sample,
            "num_comments": num_comments,
            "flat_input_ids": text_inputs["input_ids"],
            "flat_attention_mask": text_inputs["attention_mask"],
            "genre_ids": genre_ids,
            "mood_ids": mood_ids,
            "raw_genres": raw_genres,
            "raw_moods": raw_moods,
            "num_audio_chunks": num_audio_units,
        }

        for k, v in audio_inputs.items():
            if k == "attention_mask":
                out["audio_attention_mask"] = v
            else:
                out[k] = v

        return out


# =========================================================
# ===================== pooling ========================
# =========================================================

class CommentAttentionPooling(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=0)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=0)
        return pooled


class AudioChunkAttentionPooling(nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=0)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=0)
        return pooled


# =========================================================
# ===================== tag net =======================
# =========================================================

class MultiLabelMeanEmbedding(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.empty_embedding = nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.empty_embedding, std=0.02)

    def forward(self, batch_ids: List[List[int]], device: torch.device) -> torch.Tensor:
        outs = []
        for ids in batch_ids:
            if len(ids) == 0:
                outs.append(self.empty_embedding)
            else:
                idx = torch.tensor(ids, dtype=torch.long, device=device)
                emb = self.embedding(idx)
                outs.append(emb.mean(dim=0))
        return torch.stack(outs, dim=0)


class TagEncoder(nn.Module):
    def __init__(
        self,
        num_genres: int,
        num_moods: int,
        tag_emb_dim: int = 16,
        proj_hidden_dim: int = 128,
        out_dim: int = 512,
    ):
        super().__init__()
        self.genre_encoder = MultiLabelMeanEmbedding(num_genres, tag_emb_dim)
        self.mood_encoder = MultiLabelMeanEmbedding(num_moods, tag_emb_dim)

        self.proj = nn.Sequential(
            nn.Linear(tag_emb_dim * 2, proj_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(proj_hidden_dim),
            nn.Linear(proj_hidden_dim, out_dim),
        )

    def forward(self, genre_ids: List[List[int]], mood_ids: List[List[int]], device: torch.device):
        genre_emb = self.genre_encoder(genre_ids, device=device)
        mood_emb = self.mood_encoder(mood_ids, device=device)
        x = torch.cat([genre_emb, mood_emb], dim=-1)
        x = self.proj(x)
        return x


# =========================================================
# ===================== Gate Fusion ====================
# =========================================================

class GatedFusion(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

    def forward(self, text_emb: torch.Tensor, tag_emb: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([text_emb, tag_emb], dim=-1)
        gate = self.gate(cat)
        fused_candidate = self.fuse(cat)
        out = gate * fused_candidate + (1.0 - gate) * text_emb
        return out


# =========================================================
# ===================== LoRA ======================
# =========================================================

def collect_lora_target_module_names(model: nn.Module, enable_text: bool, enable_audio: bool) -> List[str]:
    names = set()

    text_keys = ["text_model", "text_branch", "text"]
    audio_keys = ["audio_model", "audio_branch", "audio"]

    target_keywords = [
        "q_proj", "k_proj", "v_proj", "out_proj",
        "query", "key", "value",
        "proj", "projection",
        "dense", "fc1", "fc2",
    ]

    print("\n[Debug] candidate linear modules:")
    shown = 0

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        lname = full_name.lower()

        if any(k in lname for k in (text_keys + audio_keys)):
            if shown < 200:
                print(full_name)
                shown += 1

        in_text = enable_text and any(k in lname for k in text_keys)
        in_audio = enable_audio and any(k in lname for k in audio_keys)

        if (in_text or in_audio) and any(k in lname for k in target_keywords):
            names.add(full_name.split(".")[-1])

    return sorted(list(names))


def apply_lora_to_clap(
    clap_model: ClapModel,
    enable_text: bool = True,
    enable_audio: bool = True,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
):
    target_modules = collect_lora_target_module_names(
        clap_model,
        enable_text=enable_text,
        enable_audio=enable_audio,
    )

    if len(target_modules) == 0:
        raise ValueError("No LoRA target modules found. Please inspect model.named_modules().")

    print("[LoRA] target_modules:", target_modules)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    clap_model = get_peft_model(clap_model, lora_config)
    return clap_model


def load_pure_clap_weights(model: ClapModel, ckpt_path: str):
    print(f"[Load] loading pure ClapModel state_dict from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[Load] pure clap missing keys: {len(missing)}")
    print(f"[Load] pure clap unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[Load] sample missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[Load] sample unexpected keys:", unexpected[:20])


def extract_pure_clap_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    HuggingFace ClapModel state_dict
    """
    clap_module = model.clap

    if isinstance(clap_module, PeftModel):
        print("[Export] merging LoRA into base ClapModel...")
        merged = clap_module.merge_and_unload()
        if hasattr(merged, "state_dict"):
            state_dict = merged.state_dict()
        else:
            raise ValueError("Merged CLAP model has no state_dict().")
    else:
        print("[Export] model.clap is already a plain ClapModel.")
        state_dict = clap_module.state_dict()

    return {k: v.detach().cpu() for k, v in state_dict.items()}


def save_pure_clap_for_infer(model: nn.Module, save_path: str):

    ensure_dir(os.path.dirname(save_path))
    pure_state_dict = extract_pure_clap_state_dict(model)
    torch.save(pure_state_dict, save_path)
    print(f"[Export] pure clap for inference saved -> {save_path}")


# =========================================================
# ===================== model ========================
# =========================================================

class ClapCommentTagModel(nn.Module):
    def __init__(
        self,
        clap_model_path: str,
        num_genres: int,
        num_moods: int,
        use_tags: bool = True,
        use_lora: bool = True,
        finetune_text_encoder: bool = True,
        finetune_audio_encoder: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        common_emb_dim: int = 512,
        comment_pool_hidden_dim: int = 512,
        audio_chunk_pool_hidden_dim: int = 512,
        tag_emb_dim: int = 16,
        tag_proj_hidden_dim: int = 128,
        audio_pooling_mode: str = "chunk_mean",
    ):
        super().__init__()

        self.use_tags = use_tags
        self.common_emb_dim = common_emb_dim
        self.audio_pooling_mode = audio_pooling_mode

        self.clap = ClapModel.from_pretrained(clap_model_path)

        if USE_PURE_CLAP_CKPT and PURE_CLAP_CKPT_PATH is not None:
            load_pure_clap_weights(self.clap, PURE_CLAP_CKPT_PATH)
        else:
            print(f"[Load] using Hugging Face pretrained CLAP only: {clap_model_path}")

        if use_lora and (finetune_text_encoder or finetune_audio_encoder):
            self.clap = apply_lora_to_clap(
                self.clap,
                enable_text=finetune_text_encoder,
                enable_audio=finetune_audio_encoder,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        for name, p in self.clap.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.comment_pooler = CommentAttentionPooling(
            dim=common_emb_dim,
            hidden_dim=comment_pool_hidden_dim,
        )

        self.audio_chunk_pooler = AudioChunkAttentionPooling(
            dim=common_emb_dim,
            hidden_dim=audio_chunk_pool_hidden_dim,
        )

        if self.use_tags:
            self.tag_encoder = TagEncoder(
                num_genres=num_genres,
                num_moods=num_moods,
                tag_emb_dim=tag_emb_dim,
                proj_hidden_dim=tag_proj_hidden_dim,
                out_dim=common_emb_dim,
            )
            self.gate_fusion = GatedFusion(dim=common_emb_dim)
        else:
            self.tag_encoder = None
            self.gate_fusion = None

        self.text_proj = nn.Identity()
        self.audio_proj = nn.Identity()

    def get_weighted_text_embedding(
        self,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        num_comments: List[int],
    ) -> torch.Tensor:
        flat_comment_embs = self.clap.get_text_features(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        pooled_embs = []
        start = 0
        for n in num_comments:
            end = start + n
            sample_comment_embs = flat_comment_embs[start:end]
            pooled = self.comment_pooler(sample_comment_embs)
            pooled_embs.append(pooled)
            start = end

        text_emb = torch.stack(pooled_embs, dim=0)
        text_emb = self.text_proj(text_emb)
        return text_emb

    def get_audio_embedding(
        self,
        input_features: torch.Tensor,
        num_audio_chunks: List[int],
        is_longer: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_features": input_features}

        if is_longer is not None:
            kwargs["is_longer"] = is_longer
        if audio_attention_mask is not None:
            kwargs["attention_mask"] = audio_attention_mask

        flat_audio_embs = self.clap.get_audio_features(**kwargs)

        if self.audio_pooling_mode == "whole_audio_mean":
            audio_emb = flat_audio_embs
            audio_emb = self.audio_proj(audio_emb)
            return audio_emb

        pooled_audio_embs = []
        start = 0
        for n in num_audio_chunks:
            end = start + n
            sample_chunk_embs = flat_audio_embs[start:end]

            if self.audio_pooling_mode == "chunk_mean":
                pooled = sample_chunk_embs.mean(dim=0)
            elif self.audio_pooling_mode == "chunk_attention":
                pooled = self.audio_chunk_pooler(sample_chunk_embs)
            else:
                raise ValueError(f"Unsupported audio pooling mode in model: {self.audio_pooling_mode}")

            pooled_audio_embs.append(pooled)
            start = end

        audio_emb = torch.stack(pooled_audio_embs, dim=0)
        audio_emb = self.audio_proj(audio_emb)
        return audio_emb

    def forward(
        self,
        flat_input_ids: torch.Tensor,
        flat_attention_mask: torch.Tensor,
        num_comments: List[int],
        input_features: torch.Tensor,
        num_audio_chunks: List[int],
        is_longer: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        genre_ids: Optional[List[List[int]]] = None,
        mood_ids: Optional[List[List[int]]] = None,
    ):
        device = input_features.device

        text_emb = self.get_weighted_text_embedding(
            flat_input_ids=flat_input_ids,
            flat_attention_mask=flat_attention_mask,
            num_comments=num_comments,
        )

        audio_emb = self.get_audio_embedding(
            input_features=input_features,
            num_audio_chunks=num_audio_chunks,
            is_longer=is_longer,
            audio_attention_mask=audio_attention_mask,
        )

        if self.use_tags:
            tag_emb = self.tag_encoder(
                genre_ids=genre_ids,
                mood_ids=mood_ids,
                device=device,
            )
            fused_text_emb = self.gate_fusion(text_emb, tag_emb)
        else:
            tag_emb = None
            fused_text_emb = text_emb

        audio_emb = l2norm(audio_emb)
        text_emb = l2norm(text_emb)
        if tag_emb is not None:
            tag_emb = l2norm(tag_emb)
        fused_text_emb = l2norm(fused_text_emb)

        return {
            "audio_emb": audio_emb,
            "text_emb": text_emb,
            "tag_emb": tag_emb,
            "fused_text_emb": fused_text_emb,
        }


# =========================================================
# ===================== Loss / Metrics ================
# =========================================================

def get_logit_scale_from_model(model: nn.Module) -> torch.Tensor:
    clap = model.clap
    if hasattr(clap, "logit_scale_a"):
        return clap.logit_scale_a
    if hasattr(clap, "logit_scale"):
        return clap.logit_scale
    if hasattr(clap, "base_model"):
        base = clap.base_model
        if hasattr(base, "logit_scale_a"):
            return base.logit_scale_a
        if hasattr(base, "logit_scale"):
            return base.logit_scale
    raise AttributeError("Cannot find logit_scale or logit_scale_a in the CLAP model.")


def contrastive_loss(audio_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: torch.Tensor):
    logits_per_audio = logit_scale.exp() * (audio_emb @ text_emb.t())
    logits_per_text = logits_per_audio.t()

    labels = torch.arange(audio_emb.size(0), device=audio_emb.device)

    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_a + loss_t) / 2.0
    return loss, logits_per_audio, logits_per_text


def compute_multitask_losses(
    outputs: Dict[str, torch.Tensor],
    logit_scale: torch.Tensor,
    use_tags: bool = True,
) -> Dict[str, torch.Tensor]:
    loss_af, logits_af_a2t, logits_af_t2a = contrastive_loss(
        audio_emb=outputs["audio_emb"],
        text_emb=outputs["fused_text_emb"],
        logit_scale=logit_scale,
    )

    loss_at, _, _ = contrastive_loss(
        audio_emb=outputs["audio_emb"],
        text_emb=outputs["text_emb"],
        logit_scale=logit_scale,
    )

    if use_tags and outputs["tag_emb"] is not None:
        loss_ag, _, _ = contrastive_loss(
            audio_emb=outputs["audio_emb"],
            text_emb=outputs["tag_emb"],
            logit_scale=logit_scale,
        )
    else:
        loss_ag = torch.zeros_like(loss_af)

    total_loss = (
        LOSS_WEIGHT_AUDIO_FUSED * loss_af
        + LOSS_WEIGHT_AUDIO_TEXT * loss_at
        + LOSS_WEIGHT_AUDIO_TAG * loss_ag
    )

    return {
        "total_loss": total_loss,
        "loss_audio_fused": loss_af,
        "loss_audio_text": loss_at,
        "loss_audio_tag": loss_ag,
        "logits_audio_to_fused": logits_af_a2t,
        "logits_fused_to_audio": logits_af_t2a,
    }


def recall_at_k_from_logits(logits: torch.Tensor, ks: List[int]) -> Dict[str, float]:
    targets = torch.arange(logits.size(0), device=logits.device)
    sorted_idx = torch.argsort(logits, dim=1, descending=True)

    metrics = {}
    for k in ks:
        topk = sorted_idx[:, :min(k, logits.size(1))]
        hit = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"R@{k}"] = hit
    return metrics


def compute_retrieval_metrics_from_logits(
    logits_a2t: torch.Tensor,
    logits_t2a: torch.Tensor,
    ks: List[int],
) -> Dict[str, float]:
    a2t = recall_at_k_from_logits(logits_a2t, ks)
    t2a = recall_at_k_from_logits(logits_t2a, ks)

    out = {}
    for k in ks:
        out[f"a2t_R@{k}"] = a2t[f"R@{k}"]
        out[f"t2a_R@{k}"] = t2a[f"R@{k}"]
        out[f"mean_R@{k}"] = (out[f"a2t_R@{k}"] + out[f"t2a_R@{k}"]) / 2.0
    return out


# =========================================================
# ===================== optimizer ========================
# =========================================================

def build_optimizer(model: nn.Module):
    new_module_params = []
    lora_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(p)
        else:
            new_module_params.append(p)

    param_groups = []
    if len(new_module_params) > 0:
        param_groups.append({
            "params": new_module_params,
            "lr": LR_NEW_MODULES,
            "weight_decay": WEIGHT_DECAY,
        })
    if len(lora_params) > 0:
        param_groups.append({
            "params": lora_params,
            "lr": LR_LORA,
            "weight_decay": WEIGHT_DECAY,
        })

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


# =========================================================
# ===================== train =====================
# =========================================================

def forward_one_batch(model: nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    outputs = model(
        flat_input_ids=batch["flat_input_ids"],
        flat_attention_mask=batch["flat_attention_mask"],
        num_comments=batch["num_comments"],
        input_features=batch["input_features"],
        num_audio_chunks=batch["num_audio_chunks"],
        is_longer=batch.get("is_longer", None),
        audio_attention_mask=batch.get("audio_attention_mask", None),
        genre_ids=batch["genre_ids"],
        mood_ids=batch["mood_ids"],
    )

    logit_scale = get_logit_scale_from_model(model)
    loss_dict = compute_multitask_losses(
        outputs=outputs,
        logit_scale=logit_scale,
        use_tags=USE_TAGS,
    )
    return loss_dict


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    sum_loss = 0.0
    sum_loss_af = 0.0
    sum_loss_at = 0.0
    sum_loss_ag = 0.0
    steps = 0
    oom_skipped = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        try:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            loss_dict = forward_one_batch(model, batch)
            loss = loss_dict["total_loss"]

            loss.backward()

            if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            sum_loss += float(loss.item())
            sum_loss_af += float(loss_dict["loss_audio_fused"].item())
            sum_loss_at += float(loss_dict["loss_audio_text"].item())
            sum_loss_ag += float(loss_dict["loss_audio_tag"].item())
            steps += 1

            if step % PRINT_FREQ == 0:
                show_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else 0.0
                pbar.set_postfix({
                    "loss": f"{sum_loss / max(steps, 1):.4f}",
                    "af": f"{sum_loss_af / max(steps, 1):.4f}",
                    "at": f"{sum_loss_at / max(steps, 1):.4f}",
                    "ag": f"{sum_loss_ag / max(steps, 1):.4f}",
                    "oom_skip": oom_skipped,
                    "lr": f"{show_lr:.2e}",
                })

        except RuntimeError as e:
            if SKIP_CUDA_OOM and is_cuda_oom_error(e):
                oom_skipped += 1
                if oom_skipped <= MAX_OOM_WARNINGS_PER_EPOCH:
                    print(f"\n[Train][OOM Skip] epoch={epoch}, step={step}, skipped={oom_skipped}")
                    print(f"[Train][OOM Skip] reason: {e}")

                optimizer.zero_grad(set_to_none=True)
                cleanup_after_oom()

                
                del batch
                continue
            raise

    if steps == 0:
        return {
            "loss": 0.0,
            "loss_audio_fused": 0.0,
            "loss_audio_text": 0.0,
            "loss_audio_tag": 0.0,
            "oom_skipped": float(oom_skipped),
        }

    return {
        "loss": sum_loss / steps,
        "loss_audio_fused": sum_loss_af / steps,
        "loss_audio_text": sum_loss_at / steps,
        "loss_audio_tag": sum_loss_ag / steps,
        "oom_skipped": float(oom_skipped),
    }

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    sum_loss = 0.0
    sum_loss_af = 0.0
    sum_loss_at = 0.0
    sum_loss_ag = 0.0
    steps = 0
    oom_skipped = 0

    metric_sums = {f"a2t_R@{k}": 0.0 for k in RETRIEVAL_KS}
    metric_sums.update({f"t2a_R@{k}": 0.0 for k in RETRIEVAL_KS})
    metric_sums.update({f"mean_R@{k}": 0.0 for k in RETRIEVAL_KS})

    pbar = tqdm(loader, desc=f"Eval Epoch {epoch}")
    for step, batch in enumerate(pbar, start=1):
        try:
            batch = move_batch_to_device(batch, device)
            loss_dict = forward_one_batch(model, batch)

            sum_loss += float(loss_dict["total_loss"].item())
            sum_loss_af += float(loss_dict["loss_audio_fused"].item())
            sum_loss_at += float(loss_dict["loss_audio_text"].item())
            sum_loss_ag += float(loss_dict["loss_audio_tag"].item())
            steps += 1

            retrieval = compute_retrieval_metrics_from_logits(
                logits_a2t=loss_dict["logits_audio_to_fused"],
                logits_t2a=loss_dict["logits_fused_to_audio"],
                ks=RETRIEVAL_KS,
            )
            for k, v in retrieval.items():
                metric_sums[k] += float(v)

        except RuntimeError as e:
            if SKIP_CUDA_OOM and is_cuda_oom_error(e):
                oom_skipped += 1
                if oom_skipped <= MAX_OOM_WARNINGS_PER_EPOCH:
                    print(f"\n[Eval][OOM Skip] epoch={epoch}, step={step}, skipped={oom_skipped}")
                    print(f"[Eval][OOM Skip] reason: {e}")

                cleanup_after_oom()
                del batch
                continue
            raise

    if steps == 0:
        out = {
            "loss": 0.0,
            "loss_audio_fused": 0.0,
            "loss_audio_text": 0.0,
            "loss_audio_tag": 0.0,
            "oom_skipped": float(oom_skipped),
        }
        for k in RETRIEVAL_KS:
            out[f"a2t_R@{k}"] = 0.0
            out[f"t2a_R@{k}"] = 0.0
            out[f"mean_R@{k}"] = 0.0
        return out

    out = {
        "loss": sum_loss / steps,
        "loss_audio_fused": sum_loss_af / steps,
        "loss_audio_text": sum_loss_at / steps,
        "loss_audio_tag": sum_loss_ag / steps,
        "oom_skipped": float(oom_skipped),
    }
    for k in RETRIEVAL_KS:
        out[f"a2t_R@{k}"] = metric_sums[f"a2t_R@{k}"] / steps
        out[f"t2a_R@{k}"] = metric_sums[f"t2a_R@{k}"] / steps
        out[f"mean_R@{k}"] = metric_sums[f"mean_R@{k}"] / steps

    return out




def build_runtime_config() -> Dict[str, Any]:
    cfg = {
        "BASE_MODEL": BASE_MODEL,
        "USE_PURE_CLAP_CKPT": USE_PURE_CLAP_CKPT,
        "PURE_CLAP_CKPT_PATH": PURE_CLAP_CKPT_PATH,
        "CSV_PATH": CSV_PATH,
        "AUDIO_DIR": AUDIO_DIR,
        "SAVE_DIR": SAVE_DIR,
        "ID_COL": ID_COL,
        "AUDIO_EXT": AUDIO_EXT,
        "COMMENT_PREFIX": COMMENT_PREFIX,
        "GENRE_PREFIX": GENRE_PREFIX,
        "MOOD_PREFIX": MOOD_PREFIX,
        "SAMPLE_RATE": SAMPLE_RATE,
        "SEED": SEED,
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "TRAIN_RATIO": TRAIN_RATIO,
        "PRINT_FREQ": PRINT_FREQ,
        "SAVE_EVERY_EPOCH": SAVE_EVERY_EPOCH,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR_NEW_MODULES": LR_NEW_MODULES,
        "LR_LORA": LR_LORA,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
        "USE_TAGS": USE_TAGS,
        "FINETUNE_TEXT_ENCODER": FINETUNE_TEXT_ENCODER,
        "FINETUNE_AUDIO_ENCODER": FINETUNE_AUDIO_ENCODER,
        "LOSS_WEIGHT_AUDIO_FUSED": LOSS_WEIGHT_AUDIO_FUSED,
        "LOSS_WEIGHT_AUDIO_TEXT": LOSS_WEIGHT_AUDIO_TEXT,
        "LOSS_WEIGHT_AUDIO_TAG": LOSS_WEIGHT_AUDIO_TAG,
        "AUDIO_POOLING_MODE": AUDIO_POOLING_MODE,
        "CHUNK_SECONDS": CHUNK_SECONDS,
        "DROP_LAST_CHUNK_IF_SHORT": DROP_LAST_CHUNK_IF_SHORT,
        "PAD_SHORT_AUDIO_TO_CHUNK": PAD_SHORT_AUDIO_TO_CHUNK,
        "WHOLE_AUDIO_MAX_SECONDS": WHOLE_AUDIO_MAX_SECONDS,
        "USE_LORA": USE_LORA,
        "LORA_R": LORA_R,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "COMMON_EMB_DIM": COMMON_EMB_DIM,
        "COMMENT_POOL_HIDDEN_DIM": COMMENT_POOL_HIDDEN_DIM,
        "AUDIO_CHUNK_POOL_HIDDEN_DIM": AUDIO_CHUNK_POOL_HIDDEN_DIM,
        "TAG_EMB_DIM": TAG_EMB_DIM,
        "TAG_PROJ_HIDDEN_DIM": TAG_PROJ_HIDDEN_DIM,
        "USE_JSON_VOCAB": USE_JSON_VOCAB,
        "GENRE_VOCAB_JSON": GENRE_VOCAB_JSON,
        "MOOD_VOCAB_JSON": MOOD_VOCAB_JSON,
        "GENRE_LABELS": GENRE_LABELS,
        "MOOD_LABELS": MOOD_LABELS,
        "RETRIEVAL_KS": RETRIEVAL_KS,
    }
    return cfg


def save_checkpoint_and_config(
    save_dir: str,
    ckpt_name: str,
    config_name: str,
    checkpoint: Dict[str, Any],
    config: Dict[str, Any],
):
    ckpt_path = os.path.join(save_dir, ckpt_name)
    cfg_path = os.path.join(save_dir, config_name)
    torch.save(checkpoint, ckpt_path)
    save_json(config, cfg_path)
    print(f"[Save] checkpoint -> {ckpt_path}")
    print(f"[Save] config     -> {cfg_path}")


def save_train_checkpoint_bundle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    best_val_loss_so_far: float,
    genre_vocab: Dict[str, int],
    mood_vocab: Dict[str, int],
    runtime_config: Dict[str, Any],
    save_dir: str,
    ckpt_name: str,
    config_name: str,
    export_pure_clap_name: Optional[str] = None,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),              
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss_so_far": best_val_loss_so_far,
        "genre_vocab": genre_vocab,
        "mood_vocab": mood_vocab,
    }

    cfg = deepcopy(runtime_config)
    cfg["epoch"] = epoch
    cfg["train_metrics"] = {k: float(v) for k, v in train_metrics.items()}
    cfg["val_metrics"] = {k: float(v) for k, v in val_metrics.items()}
    cfg["best_val_loss_so_far"] = float(best_val_loss_so_far)

    save_checkpoint_and_config(
        save_dir=save_dir,
        ckpt_name=ckpt_name,
        config_name=config_name,
        checkpoint=checkpoint,
        config=cfg,
    )

    if export_pure_clap_name is not None:
        save_pure_clap_for_infer(
            model=model,
            save_path=os.path.join(save_dir, export_pure_clap_name),
        )


# =========================================================
# ===================== main ==========================
# =========================================================

def main():
    set_seed(SEED)
    ensure_dir(SAVE_DIR)

    device = torch.device(DEVICE)
    print(f"[Info] device = {device}")

    runtime_config = build_runtime_config()
    save_json(runtime_config, os.path.join(SAVE_DIR, "run_config.json"))

    genre_vocab = load_vocab_maybe_json(USE_JSON_VOCAB, GENRE_VOCAB_JSON, GENRE_LABELS)
    mood_vocab = load_vocab_maybe_json(USE_JSON_VOCAB, MOOD_VOCAB_JSON, MOOD_LABELS)

    save_json(genre_vocab, os.path.join(SAVE_DIR, "genre_vocab.json"))
    save_json(mood_vocab, os.path.join(SAVE_DIR, "mood_vocab.json"))

    print("[Info] num_genres =", len(genre_vocab))
    print("[Info] num_moods  =", len(mood_vocab))

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print("[Info] processor sr =", processor.feature_extractor.sampling_rate)

    full_dataset = ClapTagDataset(
        csv_path=CSV_PATH,
        audio_dir=AUDIO_DIR,
        genre_vocab=genre_vocab,
        mood_vocab=mood_vocab,
        id_col=ID_COL,
        audio_ext=AUDIO_EXT,
        sample_rate=SAMPLE_RATE,
        comment_prefix=COMMENT_PREFIX,
        genre_prefix=GENRE_PREFIX,
        mood_prefix=MOOD_PREFIX,
    )

    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    collator = ClapCollator(
        processor=processor,
        sample_rate=SAMPLE_RATE,
        audio_pooling_mode=AUDIO_POOLING_MODE,
        chunk_seconds=CHUNK_SECONDS,
        drop_last_if_short=DROP_LAST_CHUNK_IF_SHORT,
        pad_short_audio_to_chunk=PAD_SHORT_AUDIO_TO_CHUNK,
        whole_audio_max_seconds=WHOLE_AUDIO_MAX_SECONDS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collator,
    )

    model = ClapCommentTagModel(
        clap_model_path=BASE_MODEL,
        num_genres=len(genre_vocab),
        num_moods=len(mood_vocab),
        use_tags=USE_TAGS,
        use_lora=USE_LORA,
        finetune_text_encoder=FINETUNE_TEXT_ENCODER,
        finetune_audio_encoder=FINETUNE_AUDIO_ENCODER,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        common_emb_dim=COMMON_EMB_DIM,
        comment_pool_hidden_dim=COMMENT_POOL_HIDDEN_DIM,
        audio_chunk_pool_hidden_dim=AUDIO_CHUNK_POOL_HIDDEN_DIM,
        tag_emb_dim=TAG_EMB_DIM,
        tag_proj_hidden_dim=TAG_PROJ_HIDDEN_DIM,
        audio_pooling_mode=AUDIO_POOLING_MODE,
    ).to(device)

    if PRINT_TRAINABLE_PARAMS:
        print_trainable_parameters(model)

    optimizer = build_optimizer(model)

    total_train_steps = EPOCHS * max(len(train_loader), 1)
    warmup_steps = int(total_train_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    print(f"[Info] total_train_steps = {total_train_steps}")
    print(f"[Info] warmup_steps      = {warmup_steps}")

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_metrics = evaluate(model, val_loader, device, epoch)

        print("=" * 100)
        print(f"[Epoch {epoch}] train_loss            = {train_metrics['loss']:.6f}")
        print(f"[Epoch {epoch}] train_loss_audio_fused = {train_metrics['loss_audio_fused']:.6f}")
        print(f"[Epoch {epoch}] train_loss_audio_text  = {train_metrics['loss_audio_text']:.6f}")
        print(f"[Epoch {epoch}] train_loss_audio_tag   = {train_metrics['loss_audio_tag']:.6f}")
        print(f"[Epoch {epoch}] train_oom_skipped      = {int(train_metrics['oom_skipped'])}")

        print(f"[Epoch {epoch}] val_loss              = {val_metrics['loss']:.6f}")
        print(f"[Epoch {epoch}] val_loss_audio_fused  = {val_metrics['loss_audio_fused']:.6f}")
        print(f"[Epoch {epoch}] val_loss_audio_text   = {val_metrics['loss_audio_text']:.6f}")
        print(f"[Epoch {epoch}] val_loss_audio_tag    = {val_metrics['loss_audio_tag']:.6f}")
        print(f"[Epoch {epoch}] val_oom_skipped       = {int(val_metrics['oom_skipped'])}")

        for k in RETRIEVAL_KS:
            print(
                f"[Epoch {epoch}] Retrieval R@{k}: "
                f"a2t={val_metrics[f'a2t_R@{k}']:.4f} "
                f"t2a={val_metrics[f't2a_R@{k}']:.4f} "
                f"mean={val_metrics[f'mean_R@{k}']:.4f}"
            )
        print("=" * 100)

        # latest：保存整训练 checkpoint + 导出纯 clap
        save_train_checkpoint_bundle(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss_so_far=min(best_val_loss, val_metrics["loss"]),
            genre_vocab=genre_vocab,
            mood_vocab=mood_vocab,
            runtime_config=runtime_config,
            save_dir=SAVE_DIR,
            ckpt_name="latest.pt",
            config_name="latest_config.json",
            export_pure_clap_name="latest_clap_for_infer.pt",
        )

        if SAVE_EVERY_EPOCH:
            save_train_checkpoint_bundle(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss_so_far=min(best_val_loss, val_metrics["loss"]),
                genre_vocab=genre_vocab,
                mood_vocab=mood_vocab,
                runtime_config=runtime_config,
                save_dir=SAVE_DIR,
                ckpt_name=f"epoch_{epoch}.pt",
                config_name=f"epoch_{epoch}_config.json",
                export_pure_clap_name=f"epoch_{epoch}_clap_for_infer.pt",
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            best_runtime = deepcopy(runtime_config)
            best_runtime["best_epoch"] = best_epoch
            best_runtime["best_val_loss"] = float(best_val_loss)

            save_train_checkpoint_bundle(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_val_loss_so_far=best_val_loss,
                genre_vocab=genre_vocab,
                mood_vocab=mood_vocab,
                runtime_config=best_runtime,
                save_dir=SAVE_DIR,
                ckpt_name="best.pt",
                config_name="best_config.json",
                export_pure_clap_name="best_clap_for_infer.pt",
            )

            print(f"[Best] epoch={best_epoch}, val_loss={best_val_loss:.6f}")

    print("=" * 100)
    print(f"[Done] best_epoch = {best_epoch}")
    print(f"[Done] best_val_loss = {best_val_loss:.6f}")
    print(f"[Done] save_dir = {SAVE_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
