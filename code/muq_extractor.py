import os
import argparse
from pathlib import Path
from tqdm import tqdm

import librosa
import torch
from muq import MuQ


def build_muq_model(repo_id, device):
    """
    Load MuQ model from HuggingFace using the official muq package.
    """
    model = MuQ.from_pretrained(repo_id)
    model = model.to(device)
    model.eval()
    return model


def collect_audio_files(input_dir, exts=("wav",)):
    input_dir = Path(input_dir)

    files = []
    for ext in exts:
        files.extend(input_dir.rglob(f"*.{ext}"))
        files.extend(input_dir.rglob(f"*.{ext.upper()}"))

    return sorted(files)


def make_output_path(audio_path, input_dir, output_dir, suffix_mode="strip_ext"):
    audio_path = Path(audio_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    rel_path = audio_path.relative_to(input_dir)

    if suffix_mode == "strip_ext":
        out_rel = rel_path.with_suffix(".pt")
    else:
        out_rel = Path(str(rel_path) + ".pt")

    return output_dir / out_rel


def extract_one(model, audio_path, save_path, device, overwrite=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(str(save_path) + ".tmp")

    if save_path.exists() and not overwrite:
        return "skip"

    if tmp_path.exists():
        tmp_path.unlink()

    # MuQ uses 24 kHz audio
    wav, _ = librosa.load(audio_path, sr=24000, mono=True)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(wav, output_hidden_states=True)

        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, dict) and "last_hidden_state" in output:
            hidden = output["last_hidden_state"]
        else:
            raise RuntimeError(f"Cannot find last_hidden_state in model output: {type(output)}")

        # Mean pooling: [B, T, D] -> [B, D]
        emb = torch.mean(hidden, dim=1)

    emb = emb.detach().cpu()
    torch.save(emb, tmp_path)
    os.replace(tmp_path, save_path)

    return "ok"


def append_line(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="OpenMuQ/MuQ-large-msd-iter",
        help="HuggingFace MuQ repo id",
    )

    parser.add_argument(
        "--audio_ext",
        type=str,
        default="wav",
        choices=["wav", "mp3"],
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--suffix_mode",
        type=str,
        default="strip_ext",
        choices=["strip_ext", "keep_ext"],
        help="strip_ext: xxx.wav -> xxx.pt; keep_ext: xxx.wav -> xxx.wav.pt",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable, switch to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")
    print(f"MuQ repo: {args.hf_repo_id}")

    model = build_muq_model(
        repo_id=args.hf_repo_id,
        device=args.device,
    )

    audio_paths = collect_audio_files(args.input_dir)
    print(f"Found {len(audio_paths)} {args.audio_ext} files")

    failed_log = Path(args.output_dir) / "failed.txt"
    done_log = Path(args.output_dir) / "done.txt"

    num_ok = 0
    num_skip = 0
    num_fail = 0

    for audio_path in tqdm(audio_paths, desc="Extracting MuQ embeddings"):
        save_path = make_output_path(
            audio_path=audio_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            suffix_mode=args.suffix_mode,
        )

        try:
            status = extract_one(
                model=model,
                audio_path=audio_path,
                save_path=save_path,
                device=args.device,
                overwrite=args.overwrite,
            )

            if status == "ok":
                num_ok += 1
                append_line(done_log, f"{audio_path}\t{save_path}")
            elif status == "skip":
                num_skip += 1

        except Exception as e:
            num_fail += 1
            append_line(failed_log, f"[ERROR]\t{audio_path}\t{repr(e)}")

    print("\n========== 提取完成 ==========")
    print(f"成功提取: {num_ok}")
    print(f"跳过已有: {num_skip}")
    print(f"失败数量: {num_fail}")
    print(f"输出目录: {args.output_dir}")
    print(f"失败日志: {failed_log}")


if __name__ == "__main__":
    main()