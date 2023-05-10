#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PV1003q
#SBATCH -n 1
#SBATCH --nodelist=node15


module add cuda11.2/toolkit/11.2.0
source /home/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='0,1,2'
export HYDRA_FULL_ERROR=1

python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Go     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Go"



export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh


   
