nohup python train_dense_encoder.py train_datasets=[code_retrieval_cpp_train] dev_datasets=[code_retrieval_cpp_dev] train=biencoder_nq output_dir=dumped &

nohup python train_dense_encoder.py \
    train=biencoder_nq \
    output_dir=dumped_ret_xcodeeval &
    
# python train_dense_encoder.py \
# train_datasets=[nq_dev] \
# dev_datasets=[nq_dev] \
# train=biencoder_local \
# output_dir=dumped


export CUDA_VISIBLE_DEVICES='0'
python generate_dense_embeddings.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
ctx_src=dpr_code_cpp \
shard_id=0 num_shards=1 \
out_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/


export CUDA_VISIBLE_DEVICES='0'
python generate_dense_embeddings.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
ctx_src=dpr_code_cpp_mini \
shard_id=0 num_shards=1 \
out_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/

python generate_dense_embeddings.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
ctx_src=dpr_code_cpp_small \
shard_id=0 num_shards=1 \
out_file=/home/maruf/zarzis/zarzis/outputs/dpr_code_cpp_small_emb/


export CUDA_VISIBLE_DEVICES='5'
python dense_retriever.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
qa_dataset=code_retrieval_cpp_test_mini_null \
ctx_datatsets=[dpr_code_cpp_mini] \
encoded_ctx_files=[/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/_0] \
out_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/out.txt



python dense_retriever.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
qa_dataset=nq_test \
ctx_datatsets=[dpr_code_cpp_mini] \
encoded_ctx_files=[/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/_0] \
out_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/

python dense_retriever.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
qa_dataset=code_retrieval_cpp_test_null \
ctx_datatsets=[dpr_code_cpp] \
encoded_ctx_files=[/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/_0] \
out_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dense_retriever_out


python dense_retriever.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
qa_dataset=code_retrieval_cpp_test_mini_null \
ctx_datatsets=[dpr_code_cpp_small] \
encoded_ctx_files=[/home/maruf/zarzis/zarzis/outputs/dpr_code_cpp_small_emb/cpp_kng_small_bin] \
out_file=/home/maruf/zarzis/zarzis/outputs/drout.out

python generate_dense_embeddings.py \
model_file=/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dumped/dpr_biencoder.39 \
ctx_src=dpr_code_cpp \
shard_id=0 num_shards=1 \
out_file=/home/maruf/zarzis/zarzis/outputs/cpp_kng_base_emb

python dense_retriever_server.py \
model_file=/home/zarzis/code/python/zarzis/dpr_deps/dpr_biencoder.39 \
qa_dataset=code_retrieval_cpp_test_null \
ctx_datatsets=[dpr_code_cpp] \
encoded_ctx_files=[/home/zarzis/code/python/zarzis/dpr_deps/_0] \
out_file=/home/zarzis/code/python/zarzis/dpr_deps/dense_retriever_out

python generate_dense_embeddings.py \
model_file=/home/zarzis/code/python/zarzis/dpr_deps/dpr_biencoder.39 \
ctx_src=dpr_code_cpp_small \
shard_id=0 num_shards=1 \
out_file=/home/maruf/zarzis/zarzis/outputs/
