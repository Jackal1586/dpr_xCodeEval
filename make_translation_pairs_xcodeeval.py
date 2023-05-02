import argparse
import os

import jsonlines
from tqdm import tqdm

def get_file_name(out_dir, lang):
	return os.path.join(out_dir, f"lang.jsonl")

def main(train_data_path, out_dir):
    jfp_by_lang = {}

    with jsonlines.open(train_data_path) as train_data_rp:
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
    parser_main = subparsers.add_parser("main", help="Run the main function")
    parser_main.add_argument(
        "--train_data_path", type=str, help="Path to the input training data"
    )
    parser_main.add_argument(
        "--out_dir", type=str, help="Output directory path for the generated embeddings"
    )
    parser_main.set_defaults(func=main)

    return parser.parse_args()

def make_pairs(data_path, lang1, lang2):
    # make dir to store embedding
    embedding_dir = os.path.join(data_path, "emb_temp_store")
    os.makedirs(embedding_dir, exist_ok=True)
    # group codes by src_uid of lang2
    lang2_dataset_path = os.path.join(data_path, get_file_name(data_path, lang2))
    lang2_dt_by_src_uid = {}
    with jsonlines.open(lang2_dataset_path) as lang2_dt_jrp:
        for dt in tqdm(lang2_dt_jrp):
            src_uid = dt["src_uid"]
            if src_uid not in lang2_dt_by_src_uid:
                lang2_dt_by_src_uid[src_uid] = []
            lang2_dt_by_src_uid[src_uid].append(dt)
    # embed each group
    
    # iterate over data of lang1
    
    ## for given src_uid retrieve  from that split
    
    # delete embedding dir
    pass

if __name__ == "__main__":
    args = parse_args()
    args.func(args.train_data_path, args.out_dir)
