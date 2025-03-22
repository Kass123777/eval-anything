from abc import ABC, abstractmethod
from typing import Optional
from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from hashlib import sha256
from eval_anything.utils.data_type import ModalityType
import os
import re
import json
import yaml
from typing import Any, Dict, Optional, Union, List
from collections import namedtuple
from pathlib import Path
import numpy as np
import contextlib
import faulthandler
import io
import multiprocessing
import platform
import signal
import tempfile
import itertools 
import base64
import PIL
from PIL import Image
from io import BytesIO

BENCHMARK_MODALITY_MAP = {
    'gsm8k': 'text_to_text',
    'mmlu': 'text_to_text',
    'truthfulqa': 'text_to_text',
    'arc': 'text_to_text',
    'cmmlu': 'text_to_text',
    'mmlupro': 'text_to_text',
    'ceval': 'text_to_text',
    'humaneval': 'text_to_text',
    'agieval': 'text_to_text',
    'beavertails': 'text_to_text',
    'mmmu': 'text_image_to_text',
    'mmau': 'text_audio_to_text',
    'mmvu': 'text_video_to_text',
}

class MultiChoicePromptBuilder():
    def __init__(self, candidate_labels: list[str], multi_choice_prompt: Optional[str] = None, cot_context: Optional[str] = None, few_shot_examples: Optional[list[str]] = None, cot: bool = False):
        self.multi_choice_prompt = multi_choice_prompt if multi_choice_prompt else "Now please answer the following multiple choice question."
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = "") -> str:
        # Handle the ground_truth label mapping
        if ground_truth.isdigit():
            ground_truth = [self.candidate_labels[int(ground_truth)]]
        prompt = f"{question}\n"
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f"({label}) {answer} "

        if ground_truth:
            answer = f"\nAnswer: ({ground_truth})"
        else:
            answer = ""

        return prompt + answer + "\n"

    def build_prompt(self, question: str, data_item: dict, question_key: str = "question", answer_key: Union[tuple, list, str] = "choices", ground_truth_key: str = "answer") -> str:
        prompt = ""

        if self.few_shot_examples:
            # Add few-shot examples
            prompt += "The following are multiple choice questions with answers.\n"
            for q, c, a in zip(self.few_shot_examples[question_key], 
                            self.few_shot_examples[answer_key],
                            self.few_shot_examples[ground_truth_key]):
                prompt += self.marge_QA(q, c, str(a)) + "\n"

        prompt += f"{self.multi_choice_prompt}\n\n"

        if isinstance(answer_key, tuple):
            answer_key = answer_key._asdict()
            value = data_item
            for key in answer_key.keys():
                value = value[key]
            candidate_answers = value
        elif isinstance(answer_key, list):
            candidate_answers = [data_item[answer_key_single] for answer_key_single in answer_key]
        else:
            candidate_answers = data_item[answer_key]

        # Add the current question
        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f"\n{self.cot_context}"
        prompt += "Please enclose your answer in parentheses. For example, (A) or (B) or (C) or (D)."
        return prompt
    
class MultiChoiceAutoLabelPromptBuilder():
    def __init__(self, multi_choice_prompt: Optional[str] = None, cot_context: Optional[str] = None, few_shot_examples: Optional[list[str]] = None, cot: bool = False):
        self.multi_choice_prompt = multi_choice_prompt if multi_choice_prompt else "Now please answer the following multiple choice question."
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = "") -> str:
        candidate_labels = [chr(65 + i) for i in range(len(candidate_answers))]

        if ground_truth.isdigit():
            ground_truth = [candidate_labels[int(ground_truth)]]
        prompt = f"{question}\n"
        for label, answer in zip(candidate_labels, candidate_answers):
            prompt += f"({label}) {answer} "

        if ground_truth:
            answer = f"\nAnswer: ({ground_truth})"
        else:
            answer = ""

        return prompt + answer + "\n"

    def build_prompt(self, question: str, candidate_answers: list[str]) -> str:
        prompt = ""

        if self.few_shot_examples:
            prompt += "The following are multiple choice questions with answers.\n"
            for q, c, a in zip(self.few_shot_examples['question'], 
                             self.few_shot_examples['choices'],
                             self.few_shot_examples['answer']):
                prompt += self.marge_QA(q, c, str(a))

        prompt += f"{self.multi_choice_prompt}\n\n"

        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f"\n{self.cot_context}"
        return prompt

class MultiChoicePromptChineseBuilder():
    def __init__(self, candidate_labels: list[str], multi_choice_prompt: Optional[str] = None, cot_context: Optional[str] = None, few_shot_examples: Optional[list[str]] = None, cot: bool = False):
        self.multi_choice_prompt = multi_choice_prompt if multi_choice_prompt else "现在请回答下面的选择题。"
        self.candidate_labels = candidate_labels
        self.cot_context = cot_context if cot_context else "让我们一步一步来思考。"
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, candidate_answers: list[str], ground_truth: str = "") -> str:
        # Handle the ground_truth label mapping
        if ground_truth.isdigit():
            ground_truth = [self.candidate_labels[int(ground_truth)]]
        prompt = f"{question}\n"
        for label, answer in zip(self.candidate_labels, candidate_answers):
            prompt += f"({label}) {answer} "

        if ground_truth:
            answer = f"\n答案: ({ground_truth})"
        else:
            answer = ""

        return prompt + answer + "\n"

    def build_prompt(self, question: str, data_item: dict, question_key: str = "question", answer_key: Union[tuple, list, str] = "choices", ground_truth_key: str = "answer") -> str:
        prompt = ""

        if self.few_shot_examples:
            prompt += "以下是带答案的多项选择题。\n"
            if isinstance(answer_key, tuple):
                answer_key = answer_key._asdict()
                answer_key_keys = list(answer_key.keys())
                for q, data_item, a in zip(self.few_shot_examples[question_key], 
                                self.few_shot_examples[answer_key_keys[0]],
                                self.few_shot_examples[ground_truth_key]):
                    value = data_item
                    for key in answer_key_keys[1:]:
                        value = value[key]
                    prompt += self.marge_QA(q, value, str(a))
            elif isinstance(answer_key, list):
                columns = [self.few_shot_examples[key] for key in answer_key]
                for q, *answers, a in zip(self.few_shot_examples[question_key],
                                *columns,
                                self.few_shot_examples[ground_truth_key]):
                    prompt += self.marge_QA(q, answers, str(a))
            else:
                for q, c, a in zip(self.few_shot_examples[question_key], 
                                self.few_shot_examples[answer_key],
                                self.few_shot_examples[ground_truth_key]):
                    prompt += self.marge_QA(q, c, str(a))

        prompt += f"{self.multi_choice_prompt}\n\n"

        if isinstance(answer_key, tuple):
            answer_key = answer_key._asdict()
            value = data_item
            for key in answer_key.keys():
                value = value[key]
            candidate_answers = value
        elif isinstance(answer_key, list):
            candidate_answers = [data_item[answer_key_single] for answer_key_single in answer_key]
        else:
            candidate_answers = data_item[answer_key]

        prompt += self.marge_QA(question, candidate_answers)
        if self.enable_cot:
            prompt += f"\n{self.cot_context}"
        return prompt

class DialoguePromptBuilder():
    def __init__(self, few_shot_examples: Optional[list[str]] = None, cot_context: Optional[str] = None, cot: bool = False):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, ground_truth: str = "") -> str:
        prompt = f"Question: {question}\n"
        answer = f"Answer: {self.cot_context} {ground_truth}" if self.enable_cot else f"Answer: {ground_truth}"
        return prompt + answer

    def build_prompt(self, input_question: str, ref_answer: str = "") -> str:
        context = ""
        if self.few_shot_examples:
            for question, answer in zip(self.few_shot_examples['question'], self.few_shot_examples['answer']):
                context += self.marge_QA(question, answer) + "\n\n"
            context = context + "\n" if context else ""

        question = self.marge_QA(input_question)
        prompt = f"{context}{question}"
        return prompt
    
class DialoguePromptChineseBuilder():
    def __init__(self, few_shot_examples: Optional[list[str]] = None, cot_context: Optional[str] = None, cot: bool = False):
        self.cot_context = cot_context if cot_context else "让我们一步一步来思考。"
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot

    def marge_QA(self, question: str, ground_truth: str = "") -> str:
        prompt = f"问题: {question}\n"
        answer = f"\n答案: {self.cot_context} {ground_truth}" if self.enable_cot else f"\n答案: {ground_truth}"
        return prompt + answer

    def build_prompt(self, input: str) -> str:
        context = ""
        if self.few_shot_examples:
            for question, answer in zip(self.few_shot_examples['question'], self.few_shot_examples['answer']):
                context += self.marge_QA(question, answer)
            context = context + "\n" if context else ""

        question = self.marge_QA(input)
        prompt = f"{context}{question}"
        return prompt

class CodesGenerationPromptBuilder():
    def __init__(self, few_shot_examples: Optional[list[str]] = None, cot_context: Optional[str] = None, cot: bool = False, language: str = "python"):
        self.cot_context = cot_context if cot_context else "Let's think step by step."
        self.few_shot_examples = few_shot_examples
        self.enable_cot = cot
        self.code_generation_prompt = 'The following are examples of function description (with Canonical_solution).'
        self.language = language
        
    def build_example_prompt(self, question: str, ground_truth: str, with_answer=True):
        answer = (
            f'Canonical_solution:\n ```{self.language}\n{ground_truth}\n```'
            if with_answer
            else ''
        )
        return f"Function description:\n{question}\n{answer}"
       
    def build_prompt(self, question: str, ground_truth: str) -> str:
        prompt = f"{self.code_generation_prompt}\n\n"
        if self.few_shot_examples:
            for few_shot_question, few_shot_ground_truth in zip(self.few_shot_examples['prompt'], self.few_shot_examples['canonical_solution']):
                prompt += self.build_example_prompt(few_shot_question, few_shot_ground_truth) + "\n"
            prompt += "Now, please provide solution for the following function description:\n"
        
        prompt += self.build_example_prompt(question, ground_truth, with_answer=False)
        prompt += f"\nPlease provide your solution in a code block using ```{self.language}\n...\n``` format."
        if self.enable_cot:
            prompt += f"\n{self.cot_context}"
        return prompt

def read_cfgs_from_yaml(yaml_relative_dir: str, yaml_name: str) -> dict[str, Any]:
    current_file_path = os.path.abspath(__file__)
    parent_path = os.path.dirname(os.path.dirname(current_file_path))
    yaml_path = os.path.join(parent_path, yaml_relative_dir, yaml_name)
    with open(yaml_path, encoding='utf-8') as f:
        try:
            configs = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc

    return configs

def update_dict(total_dict: dict[str, Any], item_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in total_dict.items():
        if key in item_dict:
            total_dict[key] = item_dict[key]
        if isinstance(value, dict):
            update_dict(value, item_dict)
    return total_dict

def is_convertible_to_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def custom_cfgs_to_dict(key_list: str, value: Any) -> dict[str, Any]:
    """This function is used to convert the custom configurations to dict."""
    if value == 'True':
        value = True
    elif value == 'False':
        value = False
    elif is_convertible_to_float(value):
        value = float(value)
    elif value.isdigit():
        value = int(value)
    elif value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
        value = value.split(',')
        value = list(filter(None, value))
    elif ',' in value:
        value = value.split(',')
        value = list(filter(None, value))
    else:
        value = str(value)
    keys_split = key_list.replace('-', '_').split(':')
    return_dict = {keys_split[-1]: value}

    for key in reversed(keys_split[:-1]):
        return_dict = {key.replace('-', '_'): return_dict}

def dict_to_namedtuple(dic):
    def convert(value):
        if isinstance(value, dict):
            return dict_to_namedtuple(value)
        elif isinstance(value, list):
            return [convert(item) for item in value]
        else:
            return value

    class EnhancedNamedTuple(namedtuple('configs', dic.keys())):
        __slots__ = ()

        def __getattr__(self, item):
            return None

    cfgs = EnhancedNamedTuple(**{k: convert(v) for k, v in dic.items()})
    return cfgs

def namedtuple_to_dict(obj):
    if hasattr(obj, '_asdict'):
        return {k: namedtuple_to_dict(v) for k, v in obj._asdict().items()}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(i) for i in obj]
    else:
        return obj

def parse_unknown_args(args):
    """
    Parse unknown arguments, support no-value parameters and return dict.
    """
    unknown_args = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                unknown_args[key] = args[i + 1]
                i += 2
            else:
                unknown_args[key] = True
                i += 1
        else:
            i += 1
    return unknown_args

def pair_data_via_uuid(inputs: list[InferenceInput], outputs: list[InferenceOutput | EvaluationResult]):
    def get_uuid_key(item: Union[InferenceInput, InferenceOutput]):
        return item.uuid
    uuid_inputs = {get_uuid_key(item): item for item in inputs}
    results = []
    for output in outputs:
        uuid_key = get_uuid_key(output)
        if uuid_key in uuid_inputs:
            results.append((uuid_inputs[uuid_key], output))
    return results

def get_messages(modality, prompt):
    messages = {
        "t2t": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "ti2t": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ],
    }
    return messages.get(modality, [])

def get_project_root():
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return None

def unsafe_execute(problem: Dict, completion: str, timeout: float, result):
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        check_program = (
            completion
            + "\n"
            + problem["test"]
            + "\n"
            + f"check({problem['entry_point']})"
        )

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(
    problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """
    # Set tokenizer parallelism before creating new process
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    manager = multiprocessing.Manager()
    result = manager.list()

    def worker_init():
        # Ensure child processes also have tokenizer parallelism disabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    p = multiprocessing.Process(target=unsafe_execute, args=(problem, completion, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    if string is None:
        return string

    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\%", "").replace(" .", " 0.").replace("{.", "{0.")

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)

    return string

def remove_few_shot_prefix(string:str):
    prefix_list = ["The answer is therefore", "答案是", "The answer is"]
    for prefix in prefix_list:
        if string.startswith(prefix):
            string = string[len(prefix):].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix):].strip()
    return string

def remove_boxed(s):
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None

    num_left_braces_open = 0
    for i in range(idx, len(s)):
        if s[i] == "{":
            num_left_braces_open += 1
        elif s[i] == "}" and num_left_braces_open and not (num_left_braces_open := num_left_braces_open - 1):
            answer = s[idx:i + 1]
            answer = answer[len("\\boxed{"):-1] if answer.startswith("\\boxed{") else answer
            return answer.split("=")[-1].lstrip(" ") if "=" in answer else answer
    return None