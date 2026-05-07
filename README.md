# madb

## Audios, annotations, and model weights

1730 audios and all annotations on our huggingface dataset: https://huggingface.co/datasets/sirui1/MADB-Dataset  
other 4400 audios from open-source muchin dataset are on https://github.com/CarlWangChina/MuChin  
model weights are on our huggingface model: https://huggingface.co/sirui1/MADB_model_v1  
we provide a sample set of 200 audios. Full samples include 200 audios and embeddings are on our huggingface dataset.
we also provide the embeddings extracted by clap, muq, clap_com_tag, and clap_com under sample folder, you can jump to evaluation step 3 with these files.  

## Prepare Environment(need GPU):

conda create -n madb python=3.11 -y  
conda activate madb  
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118  
pip install -r requirements.txt  
conda install -c conda-forge ffmpeg  #sudo apt install ffmpeg  




## Quick start with muq evaluation: 
download 'MADB_model_v1" from our huggingface model under this repo's directionary.  

for linux:   

chmod +x run_muq.bash  
./run_muq.bash | tee run.log

for windows:  

python code/to_wav.py --input_dir sample/sample_audio/ --output_dir muq/wav/ --mode muq  
python code/muq_extractor.py --input_dir muq/wav/ --output_dir muq/emb/  
python code/test.py --config code/muq_config.py --output_csv muq/result/test/test.csv



## Evaluation steps(muq, clap):
1. change to wav:  
  #mode=[muq, clap, qwen]  only decides the sample rate  
  python code/to_wav.py \
   --input_dir sample/dataset_sample/ \
   --output_dir muq/wav/   \
   --mode muq    

2. extract embeddings
   
  - python code/muq_extractor.py   \
    --input_dir muq/wav/   \
    --output_dir muq/emb/   \

  - extract clap with original weight  
  python code/extract_clap.py   \
    --input_dir clap/wav/   \
    --output_dir clap/clap/emb/   \

  - extract clap after pretrained  
  python code/extract_clap.py   \
    --input_dir clap/wav/   \
    --output_dir clap/clap_com_tag/emb/   \
    --use_infer_clap_ckpt   \
    --infer_clap_ckpt_path code/modelweight/clap_pretrain/com_tag/best_clap_for_infer.pt

   ⚠️ Notes:  

   - CLAP randomly crops long audio during extracting  
   - For reproducibility, use sample embeddings  
   - Sample embeddings:
     - 100 from Muchin  
     - 50 from Levo  
     - 50 from Suno  

3. predict score  
   python code/test.py   \
     --config muq_config.json   \
     --split all   \
     --output_csv muq_test.csv

## Evaluation steps(qwen):

!!!qwen2.5-audio instruct 7B may require different environment, please go to qwen2.5audio page for details  

python qwen_c.py --avg_csv data/annotation/song_avg_scores.csv   \
  --anno_csv data/annotation/MADB_data.csv   \
  --audio_dir data/audio/dataset   \
  --save_dir qwen/result/aud_com_tag/os/   \
  --target_col avg_overall_score   \
  --input_mode audio_comment_tag

  change --target_col to change dimension predict, and change the system prompt with the discription of dimensions in tabel 1 in paper.  
  change --input_mode  from [audio_ony, comment_only, comment_tag, audio_comment, audio_comment_tag]  


## Train steps(muq, clap without pretrain):
1. change to wav  
2. extract embeddings  
3. train with score  
   3.1 open muq_posttrain.py  
   3.2 change super parameters, which on the top of the code  
   3.3 run  
  this code train 7 dimensions without 0 values, they have seperate transformer and mlp net, and will optimize seperately.  
   3.4 open muq_train_single.py  
   3.5 change super parameters, include target dimension  
   3.6 run, and repeat 3.5 for other dimension  
 this code only train 1 dimension a time, during training, 0 calues will be ignored, so the samples are different for these 4 dimensions, arr.perc, arr.emo, sing.skill, soung.eff  

## Train steps(clap with pretrain):
1. change to wav  
2. train with commments and tags  
   2.1 open clap_pretrain.py  
   2.2 change super parameters, like USE_TAGS = False, FINETUNE_TEXT_ENCODER = False, FINETUNE_AUDIO_ENCODER = True  
   2.3 run  
3. extract embeddings with pretrained clap model weight  
4. train with score(same as before)  
    clap_posttrain.py  
    clap_train_single.py  


All posttrain codes use song_avg_score.csv, and pretrain codes use MADB_data.csv


