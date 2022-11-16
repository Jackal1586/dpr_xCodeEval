import json
from tqdm import tqdm  # type: ignore
from typing import Dict, List, Protocol, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from trie_tree import TrieTree, NodeData
from sklearn.metrics import f1_score  # type: ignore


class TextSearch(Protocol):
    def insert(self, insert_str: str, data: NodeData) -> None:
        ...

    def search(self, search_str: str) -> Optional[List[NodeData]]:
        ...


class DictAsTrieProxy:
    mapping: Dict[str, List[NodeData]]

    def __init__(self) -> None:
        self.mapping = dict()

    def insert(self, insert_str: str, data: NodeData) -> None:
        if insert_str not in self.mapping:
            self.mapping[insert_str] = list()

        self.mapping[insert_str].append(data)

    def search(self, search_str: str) -> Optional[List[NodeData]]:
        return self.mapping.get(search_str)


def load_tags_by_sub_id(raw_stats_file: str) -> Dict[str, List[str]]:
    tags_by_sub_id: Dict[str, List[str]] = {}

    with open(raw_stats_file) as raw_stats_ptr:
        for sample_str in tqdm(raw_stats_ptr, miniters=262144, desc="Reading stats"):
            sample = json.loads(sample_str)

            tags_by_sub_id[str(sample.get("submission_id"))] = sample.get("tags")

    return tags_by_sub_id


@dataclass(frozen=True)
class CFSample:
    contest_id: str
    problem_initial: str
    lang: str
    handle: str
    source_code: str
    code_size: int
    problem_id: str
    submission_id: int
    verdict: str
    tokens_length: int
    compilation_error: bool
    waiting: bool
    prettify_class: str
    href: str
    problem_name: str
    contest_name: str
    n_testcase: int
    code_author: str
    rating: int = -1
    tags: List[str] = field(default_factory=list)


@dataclass
class DPRContext:
    id: str
    title: str
    text: str
    score: float
    has_answer: bool

    def get_sub_id(self):
        return self.id.replace("cf_cpp:", "")


@dataclass
class DPRResult:
    question: str
    question_sub_id: int = field(init=False)
    question_tags: List[str] = field(init=False)
    answers: List[str] = field(default_factory=list)
    ctxs: List[DPRContext] = field(default_factory=list)

    def __post_init__(self):
        tmp = self.ctxs.copy()

        self.ctxs = [DPRContext(**ctx) for ctx in tmp]

    def validate(self):
        return self.question in [ctx.text for ctx in self.ctxs]


def load_dense_retriever_output(
    dense_retriever_output_file: str,
    search_id_by_code: TextSearch,
    tags_by_sub_id: Dict[str, List[str]],
):
    dense_retriever_output: List[DPRResult] = []

    with open(dense_retriever_output_file) as dpo_ptr:
        results_list = json.load(dpo_ptr)
        for result_dict in tqdm(results_list, desc="Readng DPR result"):

            result = DPRResult(**result_dict)
            search_results = search_id_by_code.search(result.question)
            if search_results is None:
                # print("question text not found")
                continue
            question_id_list = [d.sub_id for d in search_results]
            tags = set()
            for qid in question_id_list:
                _tags = tags_by_sub_id[str(qid)]
                tags.update(_tags)

            if len(question_id_list) < 11:
                result.question_sub_id = max(question_id_list)
                result.question_tags = list(tags)
            else:
                # print("SHIT", problem_ids, question_id_list)
                continue

            dense_retriever_output.append(result)

            # if not result.validate():
            # 	print(result.ctxs[0].id)

    print(f"Out of {len(results_list)}, got id of {len(dense_retriever_output)}")

    return dense_retriever_output


def init_tag_vector(tags_by_sub_id) -> Callable[[List[str]], np.ndarray]:
    tag_coord_map: Dict[str, int] = {}
    for tags in tqdm(tags_by_sub_id.values(), desc="Loading tag_map"):
        for tag in tags:
            if tag not in tag_coord_map:
                tag_coord_map[tag] = len(tag_coord_map)

    def get_tag_vector_from_tags(tags: List[str]) -> np.ndarray:
        return np.array([1 if tag in tags else 0 for tag in tag_coord_map])

    return get_tag_vector_from_tags


def make_eval_matrix(
    dpr_output: List[DPRResult],
    tags_by_sub_id: Dict[str, List[str]],
    get_tag_vector_from_tags: Callable[[List[str]], np.ndarray],
) -> np.ndarray:
    eval_matrix = np.zeros((len(dpr_output), len(dpr_output[0].ctxs)))

    for result_index, dpr_result in tqdm(enumerate(dpr_output), desc="Reading DPR output"):
        question_tag_vector = get_tag_vector_from_tags(dpr_result.question_tags)
        for ctx_index, ctx in enumerate(dpr_result.ctxs):
            eval_matrix[result_index, ctx_index] = f1_score(
                question_tag_vector,
                get_tag_vector_from_tags(tags_by_sub_id[ctx.get_sub_id()]),
            )

    return eval_matrix


def eval_dpr(eval_matrix: np.ndarray) -> List:
    row_wise_cumulative_max = np.zeros(eval_matrix.shape)
    for row in range(eval_matrix.shape[0]):
        for col in range(eval_matrix.shape[1]):
            row_wise_cumulative_max[row, col] = eval_matrix[row, 0 : col + 1].max()

    # row_wise_cumulative_max.sum(axis=0)

    return list(row_wise_cumulative_max.sum(axis=0))


def load_code_search_trie(source_files: List[str]) -> TextSearch:
    def reader():
        for source_file in source_files:
            with open(source_file) as source_ptr:
                for line in source_ptr:
                    yield line

    trie = DictAsTrieProxy()

    for line in tqdm(reader(), miniters=262144, desc="Loading trie"):
        sample_dict = json.loads(line)
        sample = CFSample(**sample_dict)

        trie.insert(sample.source_code, NodeData(sample.submission_id))

    return trie


if __name__ == "__main__":
    raw_stats_file = "raw_stats.jsonl"
    dense_retriever_output_file = (
        "/home/maruf/zarzis/zarzis/outputs/2022-08-01/16-00-23/dense_retriever_out"
    )

    source_files_for_trie = [
        "submissions_incomplete_testset_without_testcase.jsonl",
        "submissions_complete_testset_without_testcase.jsonl",
    ]

    trie = load_code_search_trie(source_files_for_trie)

    tags_by_sub_id = load_tags_by_sub_id(raw_stats_file)

    dpr_output = load_dense_retriever_output(
        dense_retriever_output_file, trie, tags_by_sub_id
    )

    get_tag_vector_from_tags = init_tag_vector(tags_by_sub_id)

    eval_matrix = make_eval_matrix(dpr_output, tags_by_sub_id, get_tag_vector_from_tags)

    ev = eval_dpr(eval_matrix)

    eval_output_file = "dpr_eval_output.txt"
    with open(eval_output_file, "w") as eval_out_ptr:
        eval_out_ptr.write(" ".join(map(str, ev)))
