import json
from tqdm import tqdm
from typing import Dict, List, Protocol, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from trie_tree import TrieTree, NodeData


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


def load_sub_id_to_problem_id_map(
	raw_stats_file: str,
) -> Dict[str, str]:
	sub_id_to_problem_id_map: Dict[str, str] = {}
	with open(raw_stats_file) as raw_stats_ptr:
		for sample_str in tqdm(raw_stats_ptr, miniters=262144):
			sample = json.loads(sample_str)

			sub_id_to_problem_id_map[str(sample.get("submission_id"))] = sample.get(
				"problem_id"
			)

	return sub_id_to_problem_id_map

def load_tags_by_sub_id(raw_stats_file: str) -> Dict[str, List[str]]:
	tags_by_sub_id: Dict[str, List[str]] = {}

	with open(raw_stats_file) as raw_stats_ptr:
		for sample_str in tqdm(raw_stats_ptr, miniters=262144):
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
	tags: List[str] = field(init=False)

	def get_sub_id(self):
		return self.id.replace("cf_cpp:", "")


@dataclass
class DPRResult:
	question: str
	question_sub_id: int = field(init=False)
	question_tags: List[str] = field(init=False)
	assoc_problem_id: str = field(init=False)
	answers: field(default_factory=list)
	ctxs: List[DPRContext]

	def __post_init__(self):
		tmp = self.ctxs.copy()

		self.ctxs = [DPRContext(**ctx) for ctx in tmp]

	def validate(self):
		return self.question in [ctx.text for ctx in self.ctxs]


def load_dense_retriever_output(
	dense_retriever_output_file: str,
	search_id_by_code: TextSearch,
	sub_id_to_problem_id_map: Dict[str, str],
):
	dense_retriever_output: List[DPRResult] = []

	with open(dense_retriever_output_file) as dpo_ptr:
		results_list = json.load(dpo_ptr)
		for result_dict in tqdm(results_list):

			result = DPRResult(**result_dict)
			search_results = search_id_by_code.search(result.question)
			if search_results is None:
				# print("question text not found")
				continue
			question_id_list = [d.sub_id for d in search_results]
			problem_ids = set()
			for qid in question_id_list:
				_problem_id = sub_id_to_problem_id_map[str(qid)]
				problem_ids.add(
					_problem_id[:-1] if _problem_id[-1].isdigit() else _problem_id
				)

			if len(problem_ids) == 1:
				result.question_sub_id = max(question_id_list)
				result.assoc_problem_id = list(problem_ids)[0]
			elif len(problem_ids) == 2:
				result.question_sub_id = max(question_id_list)
				result.assoc_problem_id = sub_id_to_problem_id_map[
					str(result.question_sub_id)
				]
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
	dpr_output: List[DPRResult], sub_id_to_problem_id_map: Dict[str, str]
) -> List[List[int]]:
	eval_matrix: List[List[int]] = list()

	for dpr_result in dpr_output:

		eval_matrix.append(
			[
				int(
					sub_id_to_problem_id_map[str(dpr_result.question_sub_id)]
					== sub_id_to_problem_id_map[ctx.id.replace("cf_cpp:", "")]
				)
				for ctx in dpr_result.ctxs
			]
		)

	return eval_matrix


def eval_dpr(eval_matrix: List[List[int]]) -> List:
	eval_matrix_np = np.array(eval_matrix)

	den = np.ones(eval_matrix_np.shape).sum(axis=0)
	neu = eval_matrix_np.sum(axis=0)

	ev = neu / den

	return ev


def load_code_search_trie(source_files: List[str]) -> TextSearch:
	def reader():
		for source_file in source_files:
			with open(source_file) as source_ptr:
				for line in source_ptr:
					yield line

	trie = DictAsTrieProxy()

	for line in tqdm(reader(), miniters=262144):
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

	sub_id_to_problem_id_map = load_sub_id_to_problem_id_map(
		raw_stats_file
	)

	tags_by_sub_id = load_tags_by_sub_id(raw_stats_file)

	dpr_output = load_dense_retriever_output(
		dense_retriever_output_file, trie, sub_id_to_problem_id_map
	)

	get_tag_vector_from_tags = init_tag_vector(tags_by_sub_id)

	eval_matrix = make_eval_matrix(dpr_output, sub_id_to_problem_id_map)

	ev = eval_dpr(eval_matrix)

	eval_output_file = "dpr_eval_output.txt"
	with open(eval_output_file, "w") as eval_out_ptr:
		eval_out_ptr.write(" ".join(map(str, ev)))
