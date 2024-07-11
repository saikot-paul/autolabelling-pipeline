#!/bin/bash

cd $SCRATCH/llama_label_generation/
export MPLCONFIGDIR=$SCRATCH 
export TOKENIZERS_PARALLELISM=true
source /opt/conda/etc/profile.d/conda.sh 
conda activate base

file=$1
dir=$2
user_prompt=$3
results_dir="./results"
output_dir="${results_dir}/${dir}/${file}"
inter_file_path="${results_dir}/benchmark/${file}/${file}_inter.csv"
cove_one="${output_dir}/${file}_cove_one.csv"

torchrun --nproc_per_node 1 stage_two.py \
--ckpt_dir Meta-Llama-3-8B-Instruct/ \
--tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
--max_seq_len 7500 \
--max_batch_size 10 \
--file_path ${inter_file_path} \
--output_path ${cove_one} \
--temperature 0.0 \
--sys_id 0 \
--user_id ${user_prompt} \
--cove_id 0 \
--label_col CoVe_Q1
