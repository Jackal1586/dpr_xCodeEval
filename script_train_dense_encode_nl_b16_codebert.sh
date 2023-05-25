#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p DGXq
#SBATCH -n 1
#SBATCH --nodelist=node19

module add cuda11.2/toolkit/11.2.0
source /home/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export HYDRA_FULL_ERROR=1
python train_dense_encoder.py \
    encoder=hf_bert
    train=biencoder_nq_b16 \
    output_dir=dumped_ret_xcodeeval 

export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh


   
