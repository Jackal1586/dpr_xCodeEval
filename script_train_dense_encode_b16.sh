#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PA100q
#SBATCH -n 1
#SBATCH --nodelist=node02

module load cuda11.6/toolkit/11.6.0
source /export/home2/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='0,2'

python train_dense_encoder.py \
    train=biencoder_nq_b16 \
    output_dir=dumped_ret_xcodeeval 

export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh


   
