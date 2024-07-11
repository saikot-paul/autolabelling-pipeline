#!/bin/bash

cd $SCRATCH/llama_label_generation/
export MPLCONFIGDIR=$SCRATCH
export TOKENIZERS_PARALLELISM=true
source /opt/conda/etc/profile.d/conda.sh
conda activate base

file=$1
results_dir="results/benchmark/${file}"
inter_file_path="${results_dir}/${file}_inter.csv"
ldf1="${results_dir}/${file}_cove_one.csv"
ldf2="${results_dir}/${file}_cove_two.csv"
agg_file_name="${results_dir}/${file}_agg.csv"

python aggregate_results.py --wdf_path ${inter_file_path} --df1_path ${ldf1} --df2_path ${ldf2} --agg_file_name ${agg_file_name}
