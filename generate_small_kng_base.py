import json
from dataclasses import dataclass, field
from enum import Enum

from tqdm import tqdm


class Verdicts(Enum):
    CRASHED = "CRASHED"
    OK = "OK"
    REJECTED = "REJECTED"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    CHALLENGED = "CHALLENGED"
    IDLENESS_LIMIT_EXCEEDED = "IDLENESS_LIMIT_EXCEEDED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    TESTING = "TESTING"
    WRONG_ANSWER = "WRONG_ANSWER"
    PRESENTATION_ERROR = "PRESENTATION_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class CFSample:
    contest_id: str
    problem_initial: str
    lang: str
    handle: str
    source_code: str
    code_size: int
    problem_id: str
    submission_id: int
    verdict: Verdicts
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
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.verdict, Verdicts):
            return

        self.verdict = [val for val in Verdicts if self.verdict == val.name][0]


def cf_sample_reader(read_ptr):
    for line in read_ptr:
        yield CFSample(**json.loads(line))


def make_kng_base(data_itr):
    kng_base = []
    for sample in tqdm(data_itr, miniters=123456):
        if not sample.lang.startswith("GNU C++") or sample.verdict is not Verdicts.OK:
            continue
        kng_base.append(
            {
                "id": sample.submission_id,
                "text": sample.source_code,
                "title": f"{sample.lang} {sample.prettify_class}",
            }
        )

    return kng_base


if __name__ == "__main__":
    in_file = "/home/zarzis/code/python/xCodeEval/data/submissions_complete_testset_without_testcase.jsonl"
    out_file = "cpp_kng_base_small.json"
    with open(out_file, "w") as write_ptr:
        with open(in_file) as read_ptr:
            json.dump(make_kng_base(cf_sample_reader(read_ptr)), write_ptr)
