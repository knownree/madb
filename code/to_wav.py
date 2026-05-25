import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_sr(mode: str):
    mode = mode.lower()
    if mode == "muq":
        return 24000
    elif mode == "clap":
        return 48000
    elif mode == "qwen":
        return 16000
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def convert_one(in_path: Path, out_path: Path, sr: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(out_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="ignore"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["muq", "clap", "qwen"],
    )

    parser.add_argument("--ext", type=str, default="mp3")

    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="CSV file with columns: id, name",
    )

    parser.add_argument(
        "--rename_to_id",
        action="store_true",
        default=False,
        help="Rename output wav file to id.wav",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sr = get_sr(args.mode)

    print(f"Mode: {args.mode}, SR: {sr}")

    # ==================================================
    # use CSV
    # ==================================================
    if args.csv_path is not None:
        df = pd.read_csv(args.csv_path)

        if "id" not in df.columns or "name" not in df.columns:
            raise ValueError("CSV must contain columns: id, name")

        file_map = {
            p.name: p
            for p in input_dir.rglob(f"*.{args.ext}")
        }

        print(f"CSV rows: {len(df)}")
        print(f"Found audio files: {len(file_map)}")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            file_name = str(row["name"])
            file_id = str(row["id"])

            in_path = file_map.get(file_name)

            if in_path is None:
                continue

            if args.rename_to_id:
                out_name = f"{file_id}.wav"
            else:
                out_name = Path(file_name).with_suffix(".wav").name

            out_path = output_dir / out_name

            try:
                convert_one(in_path, out_path, sr)
            except Exception:
                continue

    else:
        files = list(input_dir.rglob(f"*.{args.ext}"))

        print(f"Found {len(files)} files")

        for f in tqdm(files):
            rel_path = f.relative_to(input_dir)
            out_path = (output_dir / rel_path).with_suffix(".wav")

            try:
                convert_one(f, out_path, sr)
            except Exception:
                continue


if __name__ == "__main__":
    main()
