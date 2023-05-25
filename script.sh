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

python dense_retriever_server.py \
model_file=/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/index_n_model/dpr_biencoder.39 \
ctx_datatsets=[xcl_C_priv,xcl_C++_priv,xcl_CS_priv,xcl_D_priv,xcl_Go_priv,xcl_Haskell_priv,xcl_Java_priv,xcl_Javascript_priv,xcl_Kotlin_priv,xcl_Ocaml_priv,xcl_Pascal_priv,xcl_Perl_priv,xcl_PHP_priv,xcl_Python_priv,xcl_Ruby_priv,xcl_Rust_priv,xcl_Scala_priv] \
encoded_ctx_files=[/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_C_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_C++_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_CS_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_D_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Go_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Haskell_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Java_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Javascript_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Kotlin_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Ocaml_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Pascal_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Perl_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_PHP_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Python_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Ruby_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Rust_0,/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_Scala_0] 