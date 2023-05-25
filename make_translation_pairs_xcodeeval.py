import argparse
import os

import jsonlines
from tqdm import tqdm
from glob import glob

def get_file_name(out_dir, lang):
	return os.path.join(out_dir, f"{lang}.jsonl")

def main(train_data_path, out_dir):
	jfp_by_lang = {}
	os.makedirs(out_dir, exist_ok=True)
	for file in glob(train_data_path):
		with jsonlines.open(file) as train_data_rp:
			for data in tqdm(train_data_rp):
				lang = data["lang_cluster"]
				if lang not in jfp_by_lang:
					print(f"Open {get_file_name(out_dir, lang)}")
					jfp_by_lang[lang] = jsonlines.open(get_file_name(out_dir, lang), "w")

				jfp_by_lang[lang].write(data)

	for lang, jfp in jfp_by_lang.items():
		jfp.close()
		print(f"Close {get_file_name(out_dir, lang)}")



def parse_args():
	parser = argparse.ArgumentParser(description="Process some data.")
	subparsers = parser.add_subparsers(
		title="subcommands", description="valid subcommands", help="additional help"
	)

	# subparser for 'main' command
	parser_main = subparsers.add_parser("split_train", help="Split train data")
	parser_pair = subparsers.add_parser("make_pair", help="Make translation pair")

	parser_pair.add_argument(
		"--splits_dir", type=str, help="Path to the splitted training data"
	)
	parser_pair.add_argument(
		"--lang1", type=str, help="First lang"
	)
	parser_pair.add_argument(
		"--lang2", type=str, help="Second lang"
	)

	parser_main.add_argument(
		"--train_data_path", type=str, help="Path to the input training data"
	)
	parser_main.add_argument(
		"--out_dir", type=str, help="Output directory path for the generated embeddings"
	)
	parser_main.set_defaults(func=main)
	parser_pair.set_defaults(func=make_pairs)

	return parser.parse_args()

from omegaconf import OmegaConf, DictConfig
from hydra.experimental import compose, initialize
from generate_dense_embeddings import main_ns as encoder
from dense_retriever import main as retriever

def load_config(config_path: str = "conf", config_name: str = "gen_embs") -> DictConfig:
	with initialize(config_path=config_path):
		cfg = compose(config_name=config_name)
	return cfg

def make_cfg_gen_emb(out_file, raw_data):
	n_cfg = {
		"model_file": "bigcode/starencoder",
		"ctx_src": {
			"_target_": "dpr.data.retriever_data.XCLOnlineCtxSrc",
			"file": "",
			"raw_data": raw_data,
	   	},
		"shard_id": 0,
  		"num_shards": 1,
		"out_file": out_file
	}
	cfg = load_config()
	return OmegaConf.merge(cfg, OmegaConf.create(n_cfg))

def make_cfg_retriever(out_file, raw_data, n_docs):
	n_cfg = {
		"model_file": "bigcode/starencoder",
		"qa_dataset": {
		"_target_": "dpr.data.retriever_data.XCLOnlineQASrc",
			"file": "",
			"raw_data": raw_data,
		} ,
		# ctx_datatsets=[make_cfg_gen_emb()] \
		"encoded_ctx_files": [out_file],
		"n_docs": n_docs,
	}
	cfg = load_config(config_name="dense_retriever")
	return OmegaConf.merge(cfg, OmegaConf.create(n_cfg))

def get_emb_out_file(dir, name):
	return os.path.join(dir, f"{name}-emb.index")

def group_by_src_uid(data_path):
	data_by_src_uid = {}
	with jsonlines.open(data_path) as dt_jrp:
		for dt in tqdm(dt_jrp):
			src_uid = dt["src_uid"]
			if src_uid not in data_by_src_uid:
				data_by_src_uid[src_uid] = []
			data_by_src_uid[src_uid].append(dt)
   
	return data_by_src_uid

def make_pairs(data_path, lang1, lang2):
	# make dir to store embedding
	embedding_dir = os.path.join(data_path, "emb_temp_store")
	os.makedirs(embedding_dir, exist_ok=True)

	# group codes by src_uid of lang2
	lang2_dt_by_src_uid = group_by_src_uid(get_file_name(data_path, lang2))

	# embed each group
	for src_uid, raw_data in lang2_dt_by_src_uid.items():
		cfg = make_cfg_gen_emb(get_emb_out_file(embedding_dir, src_uid), raw_data)
		encoder(cfg)

	# group data of lang1 by src_uid
	lang1_dt_by_src_uid = group_by_src_uid(get_file_name(data_path, lang1))

	# for given src_uid retrieve  from that split
	for src_uid, raw_data in lang1_dt_by_src_uid.items():
		cfg = make_cfg_retriever(
	  		get_emb_out_file(embedding_dir, src_uid),
			raw_data,
		 	len(lang2_dt_by_src_uid[src_uid])
		)
		results = retriever(cfg)

	# run stable marriage for pairing according to rank

	# delete embedding dir
	os.rmdir(embedding_dir)

if __name__ == "__main__":
	args = parse_args()
	print(args)
	if args.func.__name__ == "main":
		args.func(args.train_data_path, args.out_dir)
	else:
	 	args.func(args.splits_dir, args.lang1, args.lang2)