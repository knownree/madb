# madb

Prepare Environment(need GPU):

conda create -n madb python=3.11 -y
conda activate madb

pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sudo apt install ffmpeg


Quick start with muq evaluation:

bash run.bash

Run Evaluation:

python test7.py \
  --config ${CONFIG} \
  --split ${SPLIT} \
  --output_csv results_${SPLIT}_seed${SEED}.csv \
  --save_predictions predictions_${SPLIT}_seed${SEED}.csv


python clap_extract.py \
  --input_dir /mnt/bn/musicevalbigai/wav_out \
  --output_dir /mnt/bn/musicevalbigai/clap_pt \
  --use_infer_clap_ckpt \
  --infer_clap_ckpt_path /mnt/bn/musicevalbigai/nips2026/pretrain/result/clap/12/best_clap_for_infer.pt

python clap_extract.py \
  --input_dir /mnt/bn/musicevalbigai/wav_out \
  --output_dir /mnt/bn/musicevalbigai/clap_pt


  python convert.py \
  --input_dir dataset \
  --output_dir wav_out \
  --mode clap

  
