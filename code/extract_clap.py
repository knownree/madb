#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import traceback
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel
import random
import numpy as np

SEED = 42



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help="input audio directory, recursively scan all wav files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output pt directory")

    parser.add_argument("--model_name", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--infer_clap_ckpt_path", type=str, default=None)

    parser.add_argument("--use_infer_clap_ckpt", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_fp16", type=str2bool, default="True")
    parser.add_argument("--save_as_dict", type=str2bool, default="True")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_broken_audio", action="store_true")
    parser.add_argument("--print_every", type=int, default=20)

    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def scan_wav_files(input_dir: str):
    input_dir = Path(input_dir)
    wavs = sorted(input_dir.rglob("*.wav"))
    return [str(p) for p in wavs]


def get_output_path(audio_path: str, output_dir: str) -> str:
    stem = Path(audio_path).stem
    return str(Path(output_dir) / f"{stem}.pt")


def load_audio_mono(audio_path: str):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.ndim != 2:
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sr


def resample_if_needed(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    )
    return resampler(waveform)


def load_infer_clap_weights(model: ClapModel, ckpt_path: str):
    print(f"[Load] loading infer ClapModel checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("[Load] use checkpoint['state_dict']")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("[Load] use checkpoint['model_state_dict']")
    else:
        state_dict = checkpoint
        print("[Load] use checkpoint directly as state_dict")

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned_state_dict[k] = v

    suspicious_keys = [
        "clap.base_model.model.",
        "lora_A",
        "lora_B",
        "base_layer",
        "comment_pooler",
        "tag_encoder",
        "gate_fusion",
    ]

    for k in cleaned_state_dict.keys():
        if any(x in k for x in suspicious_keys):
            raise ValueError(
                f"Detected non-pure-clap key: {k}\n"
                "This is probably a full training checkpoint or LoRA checkpoint. "
                "Use best_clap_for_infer.pt / latest_clap_for_infer.pt instead."
            )

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print(f"[Load] missing keys: {len(missing)}")
    print(f"[Load] unexpected keys: {len(unexpected)}")

    if missing:
        print("[Load] sample missing keys:", missing[:20])
    if unexpected:
        print("[Load] sample unexpected keys:", unexpected[:20])

    return model


@torch.no_grad()
def extract_clap_embedding_whole_audio(
    waveform,
    processor,
    model,
    device,
    use_fp16=True,
):
    if waveform.ndim != 2 or waveform.size(0) != 1:
        raise ValueError(f"Expected waveform shape [1, T], got {waveform.shape}")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    audio_np = waveform.squeeze(0).cpu().numpy()

    inputs = processor(
        audios=[audio_np],
        return_tensors="pt",
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding=True,
    )

    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    autocast_enabled = device.startswith("cuda") and use_fp16

    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        embedding = model.get_audio_features(**inputs)

    embedding = embedding.float().cpu().squeeze(0)
    return embedding


def main():
    args = parse_args()

    ensure_dir(args.output_dir)

    print("=" * 80)
    print("Loading CLAP model...")
    print(f"INPUT_DIR              : {args.input_dir}")
    print(f"OUTPUT_DIR             : {args.output_dir}")
    print(f"MODEL_NAME             : {args.model_name}")
    print(f"USE_INFER_CLAP_CKPT    : {args.use_infer_clap_ckpt}")
    print(f"INFER_CLAP_CKPT_PATH   : {args.infer_clap_ckpt_path}")
    print(f"DEVICE                 : {args.device}")
    print(f"USE_FP16               : {args.use_fp16}")
    print("=" * 80)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = ClapModel.from_pretrained(args.model_name)

    if args.use_infer_clap_ckpt:
        if args.infer_clap_ckpt_path is None:
            raise ValueError("--use_infer_clap_ckpt requires --infer_clap_ckpt_path")

        if not os.path.exists(args.infer_clap_ckpt_path):
            raise FileNotFoundError(args.infer_clap_ckpt_path)

        model = load_infer_clap_weights(model, args.infer_clap_ckpt_path)
    else:
        print("[Load] using Hugging Face pretrained weights only")

    model.eval()
    model.to(args.device)

    target_sr = processor.feature_extractor.sampling_rate
    projection_dim = getattr(model.config, "projection_dim", None)

    print(f"Processor target sampling rate: {target_sr}")
    print(f"Projection dim: {projection_dim}")

    audio_paths = scan_wav_files(args.input_dir)
    print(f"Total wav files found: {len(audio_paths)}")

    success = 0
    failed = 0
    skipped = 0

    for idx, audio_path in enumerate(tqdm(audio_paths, desc="Extracting CLAP embeddings")):
        out_path = get_output_path(audio_path, args.output_dir)

        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        try:
            waveform, sr = load_audio_mono(audio_path)
            waveform = resample_if_needed(waveform, sr, target_sr)

            embedding = extract_clap_embedding_whole_audio(
                waveform=waveform,
                processor=processor,
                model=model,
                device=args.device,
                use_fp16=args.use_fp16,
            )

            if args.save_as_dict:
                save_obj = {
                    "embedding": embedding,
                    "source_path": audio_path,
                    "model_name": args.model_name,
                    "infer_ckpt_path": args.infer_clap_ckpt_path if args.use_infer_clap_ckpt else None,
                    "target_sr": target_sr,
                    "projection_dim": projection_dim,
                    "whole_audio_once": True,
                }
            else:
                save_obj = embedding

            torch.save(save_obj, out_path)
            success += 1

        except Exception as e:
            failed += 1
            print(f"\n[Failed] {audio_path}")
            print(f"Reason: {e}")

            if not args.skip_broken_audio:
                raise

            traceback.print_exc()

        if (idx + 1) % args.print_every == 0:
            print(
                f"[Progress] {idx + 1}/{len(audio_paths)} | "
                f"success={success}, failed={failed}, skipped={skipped}"
            )

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Total   : {len(audio_paths)}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {args.output_dir}")
    print("=" * 80)



if __name__ == "__main__":
    main()