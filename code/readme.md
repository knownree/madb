## Qwen2-Audio Multimodal Ablation

We provide a Qwen2-Audio based evaluation script for testing different input modalities in music aesthetic score prediction. The script supports five isolated input settings:

| Mode | Input |
|---|---|
| `audio_only` | audio only |
| `comment_only` | human comments only |
| `comment_tag` | human comments + genre/mood tags |
| `audio_comment` | audio + human comments |
| `audio_comment_tag` | audio + human comments + genre/mood tags |

Unlike prompt-based masking, each mode is implemented with separate `if/elif` branches, so unavailable modalities are not mentioned in the prompt.

### Script

```bash
qwen2_clean_ablation_if_modes.py
Requirements
pip install torch transformers librosa pandas numpy scipy scikit-learn tqdm

The default model is:

Qwen/Qwen2-Audio-7B-Instruct
Input files

The script expects:

--avg_csv   # CSV containing average aesthetic scores
--anno_csv  # CSV containing comments, genre tags, and mood tags
--audio_dir # directory containing audio files

By default, audio files are loaded as:

<audio_dir>/<id>.wav

The target score column is:

avg_overall_score
Basic usage

Run audio-only evaluation:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_only \
  --limit 100

Run comment-only evaluation:

python qwen2_clean_ablation_if_modes.py \
  --input_mode comment_only \
  --limit 100

Run comment + tag evaluation:

python qwen2_clean_ablation_if_modes.py \
  --input_mode comment_tag \
  --limit 100

Run audio + comment evaluation:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_comment \
  --limit 100

Run audio + comment + tag evaluation:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_comment_tag \
  --limit 100
Full evaluation

Remove --limit to run on the full dataset:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_comment_tag
Resume interrupted runs

The script saves partial predictions after each sample. To resume an interrupted run:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_comment_tag \
  --resume
Audio truncation

To speed up inference or reduce memory usage, truncate audio to the first N seconds:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_only \
  --max_audio_sec 30
Range-based evaluation

Evaluate a specific subset:

python qwen2_clean_ablation_if_modes.py \
  --input_mode audio_comment_tag \
  --start 0 \
  --end 1000
Output files

Results are saved under:

<save_dir>/<input_mode>/

Each mode produces:

qwen2_<input_mode>_predictions.csv
qwen2_<input_mode>_predictions_partial.csv
qwen2_<input_mode>_metrics.csv
qwen2_<input_mode>_run_config.json

The metrics file includes:

Metric	Description
mse	mean squared error
lcc	Pearson correlation
srcc	Spearman rank correlation
krcc	Kendall tau correlation
n	number of valid evaluated samples
Notes

For audio-based modes, the script checks whether the processor actually generates audio feature tensors. If audio is not correctly passed into the model, the script raises an error instead of silently running as text-only.

For text-only modes, the script also checks that no audio features are generated, preventing modality leakage.
