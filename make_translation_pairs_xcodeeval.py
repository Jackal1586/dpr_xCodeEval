import argparse
import os

import jsonlines
from tqdm import tqdm


def main(train_data_path, out_dir):
    jfp_by_lang = {}

    def get_file_name(lang):
        return os.path.join(out_dir, f"lang.jsonl")

    with jsonlines.open(train_data_path) as train_data_rp:
        for data in tqdm(train_data_rp):
            lang = data["lang_cluster"]
            if lang not in jfp_by_lang:
                print(f"Open {get_file_name(lang)}")
                jfp_by_lang[lang] = jsonlines.open(get_file_name(lang), "w")

            jfp_by_lang[lang].write(data)

    for lang, jfp in jfp_by_lang.items():
        jfp.close()
        print(f"Close {get_file_name(lang)}")


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


if __name__ == "__main__":
    args = parse_args()
    args.func(args.train_data_path, args.out_dir)
