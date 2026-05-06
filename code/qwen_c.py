import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

warnings.filterwarnings("ignore")


VALID_INPUT_MODES = [
    "audio_only",
    "comment_only",
    "comment_tag",
    "audio_comment",
    "audio_comment_tag",
]


# =========================
# Basic utils
# =========================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def extract_score(text):
    """
    Extract a continuous score in [1, 5].
    Prefer SCORE: 3.8 / score=3.8; otherwise use the first valid number.
    """
    if text is None:
        return np.nan

    text = str(text).strip()

    patterns = [
        r"SCORE\s*[:=]\s*([1-5](?:\.\d+)?)",
        r"Score\s*[:=]\s*([1-5](?:\.\d+)?)",
        r"score\s*[:=]\s*([1-5](?:\.\d+)?)",
        r"评分\s*[:：]\s*([1-5](?:\.\d+)?)",
        r"\b([1-5](?:\.\d+)?)\b",
    ]

    for p in patterns:
        m = re.search(p, text)
        if m:
            val = safe_float(m.group(1))
            if 1.0 <= val <= 5.0:
                return val

    return np.nan


def collect_columns(df, prefixes):
    cols = []
    for c in df.columns:
        for p in prefixes:
            if re.fullmatch(rf"{re.escape(p)}\d+", c):
                cols.append(c)

    # Sort by suffix number: comment_eng1, comment_eng2, ...
    def _key(x):
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else 999999

    return sorted(cols, key=_key)


def join_nonempty_values(row, cols, max_items=None):
    vals = []
    for c in cols:
        v = row.get(c, "")
        if pd.isna(v):
            continue
        v = str(v).strip()
        if v and v.lower() not in ["nan", "none", "null"]:
            vals.append(v)
    if max_items is not None:
        vals = vals[:max_items]
    return vals


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {
            "n": len(y_true),
            "mse": np.nan,
            "lcc": np.nan,
            "srcc": np.nan,
            "krcc": np.nan,
        }

    return {
        "n": len(y_true),
        "mse": mean_squared_error(y_true, y_pred),
        "lcc": pearsonr(y_true, y_pred)[0],
        "srcc": spearmanr(y_true, y_pred)[0],
        "krcc": kendalltau(y_true, y_pred)[0],
    }


def move_to_device(inputs, device):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def has_audio_features(inputs):
    """
    Qwen2-Audio usually returns input_features / feature_attention_mask.
    Keep this permissive for different transformers versions.
    """
    audio_like_keys = [
        "input_features",
        "feature_attention_mask",
        "audio_values",
        "audio_features",
        "input_values",
    ]
    return any(k in inputs for k in audio_like_keys)


def format_comments(row, comment_cols, max_comments):
    comments = join_nonempty_values(row, comment_cols, max_items=max_comments)
    if len(comments) == 0:
        return "No comments provided."
    return "\n".join([f"- {x}" for x in comments])


def format_tags(row, genre_cols, mood_cols):
    genres = join_nonempty_values(row, genre_cols)
    moods = join_nonempty_values(row, mood_cols)

    genre_text = ", ".join(sorted(set(genres))) if len(genres) > 0 else "Unknown"
    mood_text = ", ".join(sorted(set(moods))) if len(moods) > 0 else "Unknown"

    return genre_text, mood_text


def common_scoring_instruction():
    """
    Keep the scoring instruction as identical as possible across modes.
    Do NOT mention absent modalities. This avoids condition leakage.
    """
    return """You are a strict professional music aesthetics evaluator.

Task:
Predict the overall score of the music on a continuous scale from 1 to 5.
The overall score represents a holistic evaluation of the music’s aesthetic quality, reflecting the integrated perception of multiple attributes and the overall listening experience.


Scoring rule:
- 1 = very poor
- 2 = poor
- 3 = average
- 4 = good
- 5 = excellent
- You may output decimals, such as 3.7 or 4.2.
- Do not explain.
- Output only one line in this exact format:
SCORE: <number>""".strip()


# =========================
# Clean mode-specific prompt builders
# =========================

def build_audio_only_prompt():
    return f"""{common_scoring_instruction()}

Now give the final overall aesthetic score.""".strip()


def build_comment_only_prompt(row, comment_cols, max_comments):
    comments = format_comments(row, comment_cols, max_comments)
    return f"""{common_scoring_instruction()}

Human comments:
{comments}

Now give the final overall aesthetic score.""".strip()


def build_comment_tag_prompt(row, comment_cols, genre_cols, mood_cols, max_comments):
    comments = format_comments(row, comment_cols, max_comments)
    genre_text, mood_text = format_tags(row, genre_cols, mood_cols)

    return f"""{common_scoring_instruction()}

Human comments:
{comments}

Genre tags:
{genre_text}

Mood tags:
{mood_text}

Now give the final overall aesthetic score.""".strip()


def build_audio_comment_prompt(row, comment_cols, max_comments):
    comments = format_comments(row, comment_cols, max_comments)

    return f"""{common_scoring_instruction()}

Human comments:
{comments}

Now give the final overall aesthetic score.""".strip()


def build_audio_comment_tag_prompt(row, comment_cols, genre_cols, mood_cols, max_comments):
    comments = format_comments(row, comment_cols, max_comments)
    genre_text, mood_text = format_tags(row, genre_cols, mood_cols)

    return f"""{common_scoring_instruction()}

Human comments:
{comments}

Genre tags:
{genre_text}

Mood tags:
{mood_text}

Now give the final overall aesthetic score.""".strip()


def build_mode_specific_prompt(row, comment_cols, genre_cols, mood_cols, input_mode, max_comments):
    """
    True if/elif isolation.
    Each mode has its own prompt constructor.
    No prompt says "audio is not provided" or "tags are not provided".
    """
    if input_mode == "audio_only":
        return build_audio_only_prompt()

    elif input_mode == "comment_only":
        return build_comment_only_prompt(
            row=row,
            comment_cols=comment_cols,
            max_comments=max_comments,
        )

    elif input_mode == "comment_tag":
        return build_comment_tag_prompt(
            row=row,
            comment_cols=comment_cols,
            genre_cols=genre_cols,
            mood_cols=mood_cols,
            max_comments=max_comments,
        )

    elif input_mode == "audio_comment":
        return build_audio_comment_prompt(
            row=row,
            comment_cols=comment_cols,
            max_comments=max_comments,
        )

    elif input_mode == "audio_comment_tag":
        return build_audio_comment_tag_prompt(
            row=row,
            comment_cols=comment_cols,
            genre_cols=genre_cols,
            mood_cols=mood_cols,
            max_comments=max_comments,
        )

    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")


# =========================
# Qwen2-Audio input builders
# =========================

def load_audio_array(audio_path, processor, max_audio_sec=None):
    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    wav, _ = librosa.load(str(audio_path), sr=sr, mono=True)

    if max_audio_sec is not None and max_audio_sec > 0:
        wav = wav[: int(max_audio_sec * sr)]

    return wav


def build_conversation_text(processor, content):
    conversation = [
        {
            "role": "system",
            "content": "You are a strict professional music aesthetics evaluator. Return only the requested score.",
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def processor_with_audio(processor, text, audios):
    """
    Use the official Qwen2-Audio style first: audios=audios.
    Some processor versions may warn, so we also try audio=audios.
    If no audio features appear, caller will raise an error.
    """
    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
    )

    if has_audio_features(inputs):
        return inputs

    try:
        sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
        inputs2 = processor(
            text=text,
            audio=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        if has_audio_features(inputs2):
            return inputs2
    except Exception:
        pass

    return inputs


@torch.no_grad()
def predict_one(
    model,
    processor,
    audio_path,
    row,
    comment_cols,
    genre_cols,
    mood_cols,
    input_mode,
    max_comments=10,
    max_new_tokens=32,
    max_audio_sec=None,
):
    """
    Five completely isolated branches.
    The difference across modes is not implemented by telling the model what is absent.
    Instead, each branch physically constructs different content and processor inputs.
    """

    prompt = build_mode_specific_prompt(
        row=row,
        comment_cols=comment_cols,
        genre_cols=genre_cols,
        mood_cols=mood_cols,
        input_mode=input_mode,
        max_comments=max_comments,
    )

    # ---------- 1. Audio only ----------
    if input_mode == "audio_only":
        audio = load_audio_array(audio_path, processor, max_audio_sec=max_audio_sec)
        content = [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": prompt},
        ]
        text = build_conversation_text(processor, content)
        inputs = processor_with_audio(processor, text=text, audios=[audio])

    # ---------- 2. Comment only ----------
    elif input_mode == "comment_only":
        content = [
            {"type": "text", "text": prompt},
        ]
        text = build_conversation_text(processor, content)
        inputs = processor(text=text, return_tensors="pt", padding=True)

    # ---------- 3. Comment + tag ----------
    elif input_mode == "comment_tag":
        content = [
            {"type": "text", "text": prompt},
        ]
        text = build_conversation_text(processor, content)
        inputs = processor(text=text, return_tensors="pt", padding=True)

    # ---------- 4. Audio + comment ----------
    elif input_mode == "audio_comment":
        audio = load_audio_array(audio_path, processor, max_audio_sec=max_audio_sec)
        content = [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": prompt},
        ]
        text = build_conversation_text(processor, content)
        inputs = processor_with_audio(processor, text=text, audios=[audio])

    # ---------- 5. Audio + comment + tag ----------
    elif input_mode == "audio_comment_tag":
        audio = load_audio_array(audio_path, processor, max_audio_sec=max_audio_sec)
        content = [
            {"type": "audio", "audio_url": str(audio_path)},
            {"type": "text", "text": prompt},
        ]
        text = build_conversation_text(processor, content)
        inputs = processor_with_audio(processor, text=text, audios=[audio])

    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

    # Hard safety check for audio modes.
    if input_mode in ["audio_only", "audio_comment", "audio_comment_tag"]:
        if not has_audio_features(inputs):
            raise RuntimeError(
                "Audio mode is enabled, but processor output contains no audio feature keys. "
                "The audio was not actually fed into the model. "
                "Please check transformers / Qwen2-Audio processor compatibility."
            )

    # Hard safety check for text-only modes: no audio features should exist.
    if input_mode in ["comment_only", "comment_tag"]:
        if has_audio_features(inputs):
            raise RuntimeError(
                "Text-only mode unexpectedly produced audio feature keys. "
                "This indicates mode leakage."
            )

    inputs = move_to_device(inputs, model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only newly generated tokens.
    if "input_ids" in inputs:
        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    text_out = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return text_out


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--avg_csv", default="data/annotation/song_avg_scores.csv")
    parser.add_argument("--anno_csv", default="data/annotation/MADB_data.csv")
    parser.add_argument("--audio_dir", default="data/audio/dataset")
    parser.add_argument("--save_dir", default="/qwen2a/aud_com_tag_os/")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--target_col", type=str, default="avg_overall_score")
    parser.add_argument("--audio_ext", type=str, default=".wav")

    parser.add_argument("--input_mode", type=str, default="audio_comment_tag", choices=VALID_INPUT_MODES)

    parser.add_argument("--max_comments", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--max_audio_sec",
        type=float,
        default=None,
        help="Optional: truncate audio to first N seconds. Useful for speed and OOM control.",
    )

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="If enabled, skip ids already present in the partial prediction CSV.",
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir) / args.input_mode
    save_dir.mkdir(parents=True, exist_ok=True)

    avg_df = pd.read_csv(args.avg_csv)
    anno_df = pd.read_csv(args.anno_csv)

    avg_df[args.id_col] = avg_df[args.id_col].astype(str)
    anno_df[args.id_col] = anno_df[args.id_col].astype(str)

    df = avg_df.merge(anno_df, on=args.id_col, how="left", suffixes=("", "_anno"))

    if args.start is not None or args.end is not None:
        start = 0 if args.start is None else args.start
        end = len(df) if args.end is None else args.end
        df = df.iloc[start:end]

    if args.limit is not None:
        df = df.head(args.limit)

    # Prefer English comments. If absent, use raw comment columns.
    comment_eng_cols = collect_columns(df, ["comment_eng"])
    comment_raw_cols = collect_columns(df, ["comment"])
    comment_cols = comment_eng_cols if len(comment_eng_cols) > 0 else comment_raw_cols

    genre_cols = collect_columns(df, ["genre"])
    mood_cols = collect_columns(df, ["mood"])

    print(f"[Info] samples: {len(df)}")
    print(f"[Info] input_mode: {args.input_mode}")
    print(f"[Info] comment columns: {comment_cols}")
    print(f"[Info] genre columns: {genre_cols}")
    print(f"[Info] mood columns: {mood_cols}")

    if args.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"[Load] model: {args.model_name}")

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()

    partial_path = save_dir / f"qwen2_{args.input_mode}_predictions_partial.csv"

    results = []
    done_ids = set()

    if args.resume and partial_path.exists():
        old_df = pd.read_csv(partial_path)
        results = old_df.to_dict("records")
        done_ids = set(old_df["id"].astype(str).tolist())
        print(f"[Resume] loaded {len(done_ids)} existing predictions from {partial_path}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = str(row[args.id_col])

        if args.resume and sample_id in done_ids:
            continue

        audio_path = Path(args.audio_dir) / f"{sample_id}{args.audio_ext}"
        true_score = safe_float(row.get(args.target_col, np.nan))

        item = {
            "id": sample_id,
            "name": row.get("name", ""),
            "input_mode": args.input_mode,
            "audio_path": str(audio_path),
            "true_score": true_score,
            "pred_score": np.nan,
            "raw_output": "",
            "status": "ok",
        }

        need_audio = args.input_mode in ["audio_only", "audio_comment", "audio_comment_tag"]

        if need_audio and not audio_path.exists():
            item["status"] = "missing_audio"
            results.append(item)
            pd.DataFrame(results).to_csv(partial_path, index=False)
            continue

        if not np.isfinite(true_score):
            item["status"] = "missing_label"
            results.append(item)
            pd.DataFrame(results).to_csv(partial_path, index=False)
            continue

        try:
            raw_output = predict_one(
                model=model,
                processor=processor,
                audio_path=audio_path,
                row=row,
                comment_cols=comment_cols,
                genre_cols=genre_cols,
                mood_cols=mood_cols,
                input_mode=args.input_mode,
                max_comments=args.max_comments,
                max_new_tokens=args.max_new_tokens,
                max_audio_sec=args.max_audio_sec,
            )

            pred_score = extract_score(raw_output)

            item["raw_output"] = raw_output
            item["pred_score"] = pred_score

            if not np.isfinite(pred_score):
                item["status"] = "parse_failed"

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            item["status"] = "cuda_oom"

        except Exception as e:
            item["status"] = f"error: {repr(e)}"

        results.append(item)
        pd.DataFrame(results).to_csv(partial_path, index=False)

    pred_df = pd.DataFrame(results)
    pred_path = save_dir / f"qwen2_{args.input_mode}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Robust numeric conversion.
    pred_df["true_score_num"] = pd.to_numeric(pred_df["true_score"], errors="coerce")
    pred_df["pred_score_num"] = pd.to_numeric(pred_df["pred_score"], errors="coerce")

    valid_df = pred_df[
        np.isfinite(pred_df["true_score_num"])
        & np.isfinite(pred_df["pred_score_num"])
    ].copy()

    metrics = compute_metrics(
        valid_df["true_score_num"].values,
        valid_df["pred_score_num"].values,
    )

    metrics_df = pd.DataFrame([metrics])
    metrics_path = save_dir / f"qwen2_{args.input_mode}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    config = vars(args)
    config["num_total"] = len(pred_df)
    config["num_valid"] = len(valid_df)
    config["metrics"] = metrics

    with open(save_dir / f"qwen2_{args.input_mode}_run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"Input mode:       {args.input_mode}")
    print(f"Prediction table: {pred_path}")
    print(f"Metrics table:    {metrics_path}")
    print(metrics_df)


if __name__ == "__main__":
    main()
