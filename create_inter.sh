#!/bin/bash

cd $SCRATCH/llama_label_generation/
export MPLCONFIGDIR=$SCRATCH 
export TOKENIZERS_PARALLELISM=true
source /opt/conda/etc/profile.d/conda.sh 
conda activate base

file=$1
results_dir="results/benchmark/${file}"
inter_file_path="${results_dir}/${file}_inter.csv"
dataset_file="./datasets/${file}.csv"

python stage_one.py --file_path "${dataset_file}" --embd_model_path "../distiluse-base-multilingual-cased-v2" --output_path "${inter_file_path}"

