#!/bin/bash

set -e  
set -o pipefail

echo "===== Step 1: Convert to wav ====="
python code/to_wav.py \
  --input_dir sample/dataset_sample/ \
  --output_dir muq/wav/ \
  --mode muq

echo "===== Step 2: Extract MuQ embeddings ====="
python code/muq_extractor.py \
  --input_dir muq/wav/ \
  --output_dir muq/emb/

echo "===== Step 3: Run inference ====="
python code/test.py \
  --config muq_config.json \
  --output_csv muq/result/test/test.csv

echo "===== DONE ====="
