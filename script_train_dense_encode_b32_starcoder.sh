#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PV1003q
#SBATCH -n 1
#SBATCH --nodelist=node14

module add cuda11.2/toolkit/11.2.0
source /home/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='0,1,2,3'

python train_dense_encoder.py \
    encoder=hf_starencoder
    train=biencoder_nq_b32 \
    output_dir=dumped_ret_xcodeeval 

export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh


   
