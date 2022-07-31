python train_dense_encoder.py \
train_datasets=[code_retrieval_cpp_train] \
dev_datasets=[code_retrieval_cpp_dev] \
train=biencoder_local \
output_dir=dumped


    
python train_dense_encoder.py \
train_datasets=[nq_dev] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir=dumped