from abc import ABC, abstractmethod
from typing import Dict, List
from collections import namedtuple

# Local imports
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.data_type import InferenceInput, InferenceOutput, EvaluationResult
from eval_anything.models.base_model import MODEL_MAP, CLASS_MAP
from eval_anything.evaluate_tools.t2t_tools import *
import eval_anything.evaluate_tools.t2t_tools as T2T_TOOLS
from eval_anything.evaluate_tools.metrics import MetricCalculator, OverallMetricCalculator
from eval_anything.utils.utils import (
    UUIDGenerator, read_cfgs_from_yaml, update_dict, 
    custom_cfgs_to_dict, BENCHMARK_MODALITY_MAP, pair_data_via_uuid,
    dict_to_namedtuple
)
from eval_anything.utils.cache_manager import CacheManager
import eval_anything.evaluate_tools.metrics as Metrics
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register("base")
class BaseBenchmark(ABC):

    def __init__(self, 
                 model: BaseModel, 
                 eval_cfgs: namedtuple, 
                 model_cfgs: namedtuple, 
                 infer_cfgs: namedtuple, 
                 output_path: str, 
                 cache_manager: CacheManager = None, 
                 logger: EvalLogger = None):
        self.model = model
        self.eval_cfgs = eval_cfgs
        self.model_cfgs = model_cfgs
        self.infer_cfgs = infer_cfgs
        self.output_path = output_path
        self.logger = logger
        self.cache_manager = cache_manager
        self.enable_cache = True if cache_manager else False
        self.benchmark_name = "base"
    
    def get_benchmark_cfgs(self, benchmark_name: str) -> dict:
        """Get benchmark configs from yaml file in benchmark folder
        Args:
            benchmark_name (str): benchmark name
            
        Returns:
            benchmark_cfgs (dict): benchmark configs, including
        """
        benchmark_cfgs = read_cfgs_from_yaml(yaml_relative_dir=f'benchmarks/{BENCHMARK_MODALITY_MAP[benchmark_name.lower()]}/{benchmark_name}', yaml_name=f'configs.yaml')
        benchmark_cfgs = dict_to_namedtuple(benchmark_cfgs)
        return benchmark_cfgs

    @abstractmethod
    def init_dataloader(self, eval_cfgs: namedtuple, data_cfgs: namedtuple):
        """Load evaluation dataset
        Args:
            eval_cfgs (namedtuple): evaluation configs
            data_cfgs (namedtuple): dataset configs
            logger (EvalLogger): logger

        Returns:
            dataloader (BaseDataLoader): dataloader
        """
        pass
    
    def batch_inference(self, model, input_dict: dict[str, list[InferenceInput]]) -> dict[str, list[InferenceOutput]]:
        # TODO 需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        
        input_list = []
        for task, data_list in input_dict.items():
            for data in data_list:
                data.uuid = UUIDGenerator()(data)
                data.task = task
                input_list.append(data)
        
        input_batch_size = 500
        input_data_batches = [input_list[i:i+input_batch_size] for i in range(0, len(input_list), input_batch_size)]
        inference_outputs = []
        for input_data_batch in input_data_batches:
            if self.enable_cache:
                cache_path, cache_exist = self.cache_manager.get_cache_path(
                    self.model_cfgs, 
                    self.infer_cfgs,
                    input_data_batch
                )
                if cache_exist:
                    inference_outputs.extend(self.cache_manager.load(cache_path))
                else:
                    batch_outputs = self.model_inference(model, input_data_batch)
                    inference_outputs.extend(batch_outputs)
                    self.cache_manager.save(cache_path, batch_outputs)
            else:
                inference_outputs.extend(self.model_inference(model, input_data_batch))

        outputs = {task: [] for task in input_dict.keys()}
        for output in inference_outputs:
            outputs[output.task].append(output)
        return outputs

    def model_inference(self, model, input_data: list[InferenceInput]):
        # TODO 需要支持多轮
        """Model inference. Support multi-round inference.
        Args:
            input_data (list[InferenceInput]): input data
            
        Returns:
            inference_outputs (list[InferenceOutput]): inference outputs
        """
        return model.generation(input_data)
    
    def run(self,
            task_list: list[str]) -> tuple[dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """Run benchmark
        Args:
            task_list (list[str]): task list
            
        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """
        self.logger.log('info', f'Evaluating {self.benchmark_name}...')

        dataloader = self.init_dataloader(self.eval_cfgs, self.benchmark_cfgs)
        input_data = dataloader.load_dataset(task_list)    # Input_data: list[InferenceInput]
        inference_outputs = self.batch_inference(self.model, input_data)
        ref_answers = {task: self.get_ref_answer(input_data[task], inference_outputs[task]) for task in task_list}
        evaluation_details = {}
        evaluation_results = {}
        for task in task_list:
            evaluation_details[task], evaluation_results[task] = self.calculate_metrics(self.benchmark_name, inference_outputs[task], ref_answers[task], self.benchmark_cfgs.answer_extractor, self.benchmark_cfgs.metrics)

        if len(task_list) > 1:
            overall_result = self.calculate_overall_metrics(self.benchmark_cfgs.overall_metrics, result=evaluation_results)
        else:
            overall_result = {}
        self.display_benchmark_results(self.benchmark_name, evaluation_results)
        if overall_result != {}:
            self.display_benchmark_results(self.benchmark_name, overall_result)
        self.save_benchmark_details(self.output_path, self.benchmark_name, input_data, evaluation_details)
        return evaluation_details, evaluation_results, overall_result
    
    def calculate_metrics(self, benchmark_name: str, inference_outputs: list[InferenceOutput], ref_answers: list[str], answer_extractors: list[namedtuple], metrics_list: list[dict], judge_method: str = 'judge_equal'):
        """Calculate metrics
        Args:
            benchmark_name (str): benchmark name
            inference_outputs (list[InferenceOutput]): inference outputs
            evaluate_tools (list[str]): evaluate tool list
            metrics_list (list[dict]): metrics list
            judge_method (str): judge method

        Returns:
            evaluation_details (dict[str, list[EvaluationResult]]): evaluation details
            evaluation_results (dict[str, dict[str, float]]): evaluation results
        """
        output_text = [item.response[0] for item in inference_outputs]
        extracted_results = {}
        for answer_extractor in answer_extractors:
            extracted_results[answer_extractor.name] = getattr(T2T_TOOLS, T2T_EXTRACTOR_MAP[answer_extractor.function])(**(answer_extractor.args._asdict())).apply(output_text)
        
        evaluation_details = []
        for inference_output, ref_answer, index in zip(inference_outputs, ref_answers, range(len(inference_outputs))):
            extracted_result = {extractor: extracted_results[extractor][index] for extractor in extracted_results.keys()}
            evaluation_details.append(EvaluationResult(benchmark_name, inference_output, extracted_result, ref_answer, inference_output.uuid))
        metric_calculator = MetricCalculator(metrics_list, judge_method)
        evaluation_results = metric_calculator(evaluation_details)
        return evaluation_details, evaluation_results
    
    def calculate_overall_metrics(self, overall_metrics: list[namedtuple], result: dict[str, dict[str, float]]) -> dict[str, float]:
        """Calculate overall metrics
        Args:
            overall_metrics (list[namedtuple]): overall metrics
            result (dict[str, dict[str, float]]): evaluation results
        """
        overall_metric_calculator = OverallMetricCalculator(overall_metrics)
        overall_result = overall_metric_calculator(result)
        return overall_result

    def get_ref_answer(self, input_data: list[InferenceInput], inference_outputs: list[InferenceOutput]) -> list[any]:
        """Get reference answers from input data
        Args:
            input_data (list[InferenceInput]): input data
            inference_outputs (list[InferenceOutput]): inference outputs
            
        Returns:
            ref_answers (list[any]): reference answers
        """
        paired_data = pair_data_via_uuid(input_data, inference_outputs)
        ref_answer = [item.ref_answer for item, _ in paired_data]
        return ref_answer
    
    def display_benchmark_results(self, benchmark_name: str, result: Dict[str, Dict[str, float]]):
        """Display single benchmark results in command line.
        Args:
            benchmark_name (str): benchmark name
            result (Dict[str, Dict[str, float]]): evaluation result
        """
        result_to_display = {}
        for task, task_result in result.items():
            result_to_display[task] = {}
            for metric, metric_result in task_result.items():
                metric_result = max(metric_result.values())
                result_to_display[task][metric] = metric_result
        self.logger.print_table(title=f'{benchmark_name} Benchmark', data=result_to_display, to_csv=True, csv_file_name=f'{benchmark_name}_results.csv')
        self.logger.log('info', '++++++++++++++++++++++++++++++++++++')
        self.logger.log('info', f'Benchmark: {benchmark_name}')
        self.logger.log('info', f"model_id: {self.model_cfgs.model_id},")
        self.logger.log('info', f"num_fewshot: {self.eval_cfgs.n_shot._asdict()[benchmark_name]},")
        self.logger.log('info', f"chain_of_thought: {self.eval_cfgs.cot._asdict()[benchmark_name]},")
        self.logger.log('info', '++++++++++++++++++++++++++++++++++++')
    
    @abstractmethod
    def save_benchmark_details(self, save_path: str, benchmark_name: str, inputs: Dict[str, List[InferenceInput]], results: Dict[str, List[EvaluationResult]]):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            benchmark_name (str): benchmark name
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        pass