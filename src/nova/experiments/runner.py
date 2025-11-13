import json
import logging
import shutil
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import mlflow
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from nova.baselines.base import BaselineOutputColumns, BaselineType
from nova.baselines.codeagent import CodeAgentWithTools, OrchestratorCodeAgent, SimpleCodeAgent
from nova.baselines.llm_baselines import (
    LLMOnly,
    LLMWithPythonInterpreter,
    LLMWithPythonInterpreterAndRetries,
)
from nova.baselines.tool_calling_agent import OrchestratorToolCallingAgent, ToolCallingAgentWithTools
from nova.evaluation.evaluator import EvalKey
from nova.experiments.questions_schema import BenchmarkQuestion, DataType
from nova.utils.azureml import get_aml_job_id
from nova.utils.config import save_config

logger = logging.getLogger(__name__)

RATE_LIMIT_ERROR_CODE = 429
EVALUATION_ERROR_MSG = "Error evaluating answer for question"


class BenchmarkCategory(StrEnum):
    DATA_QA = "data_qa"
    CELLULAR_QA = "cellular_qa"
    PATCH_QA = "patch_qa"
    SLIDE_QA = "slide_qa"


class ExperimentRunner:
    def __init__(self, config: DictConfig, benchmark_category: BenchmarkCategory = BenchmarkCategory.DATA_QA) -> None:
        self.config = config
        self.config_baseline = self.config.baseline
        self.config_benchmark = self.config.benchmark
        self.benchmark_category = benchmark_category
        self.baseline_type = self.config_baseline.type
        self.aml_job_id = get_aml_job_id()
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def experiment_description(self) -> str:
        model_name = self.config_baseline.llm.api_config.deployment_name
        return f"{self.baseline_type} - {model_name} - {self.benchmark_category}"

    @property
    def root_working_dir(self) -> Path:
        root_working_dir = Path(self.config.run.root_working_dir)
        if self.config.run.subfolder_working_dir:
            if self.aml_job_id:
                root_working_dir = root_working_dir / self.aml_job_id
            else:
                root_working_dir = (
                    root_working_dir
                    / self.baseline_type
                    / self.config_baseline.llm.api_config.deployment_name
                    / self.start_timestamp
                    / self.benchmark_category
                )
        root_working_dir.mkdir(parents=True, exist_ok=True)
        return root_working_dir

    @property
    def working_dir(self) -> Path:
        working_dir = self.root_working_dir / "working_dir"
        working_dir.mkdir(parents=True, exist_ok=True)
        return working_dir

    @property
    def output_dir(self) -> Path:
        output_dir = self.root_working_dir / "agent_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def root_path_to_dataset(self) -> Path:
        root_path_to_dataset = Path(self.config.dataset.root_path)
        if not root_path_to_dataset.exists():
            raise FileNotFoundError(f"Root Dataset path {root_path_to_dataset} does not exist.")
        return root_path_to_dataset

    @property
    def benchmark_category_path(self) -> Path:
        benchmark_root = Path(self.config_benchmark.root_path)
        if not benchmark_root.exists():
            raise FileNotFoundError(f"Benchmark root path {benchmark_root} does not exist.")
        return benchmark_root / self.benchmark_category

    def get_agent(self):
        match self.baseline_type:
            case BaselineType.LLM_ONLY:
                return LLMOnly(config=self.config.baseline)
            case BaselineType.LLM_WITH_PI:
                return LLMWithPythonInterpreter(config=self.config.baseline)
            case BaselineType.LLM_WITH_PI_AND_RETRIES:
                return LLMWithPythonInterpreterAndRetries(config=self.config.baseline)
            case BaselineType.CODEAGENT_WITHOUT_TOOLS:
                return SimpleCodeAgent(config=self.config.baseline)
            case (
                BaselineType.CODEAGENT_WITH_TOOLS
                | BaselineType.CODEAGENT_WITH_TOOLS_AND_PLANNING
                | BaselineType.CODEAGENT_WITH_RAG
            ):
                return CodeAgentWithTools(config=self.config.baseline)
            case BaselineType.TOOLCALLING_AGENT_WITH_TOOLS:
                return ToolCallingAgentWithTools(config=self.config.baseline)
            case BaselineType.ORCHESTRATOR_CODEAGENT:
                return OrchestratorCodeAgent(config=self.config.baseline)
            case BaselineType.ORCHESTRATOR_TOOLCALLING_AGENT:
                return OrchestratorToolCallingAgent(config=self.config.baseline)
            case _:
                raise ValueError(f"Unsupported baseline type: {self.baseline_type}")

    def read_benchmark_questions(self) -> list[BenchmarkQuestion]:
        """
        Read benchmark questions from the specified path.
        """
        benchmark_qs_path = self.benchmark_category_path / "questions.json"
        with open(benchmark_qs_path, "r") as file:
            questions_data = json.load(file)

        if not isinstance(questions_data, list):
            raise ValueError(f"Expected a list of questions, but got {type(questions_data)} instead.")
        if not all(isinstance(q, dict) for q in questions_data):
            raise ValueError("Each question should be a dictionary.")

        questions = []
        skipped_count = 0
        for i, q in enumerate(questions_data):
            try:
                question = BenchmarkQuestion(**q)
            except Exception as e:
                logger.warning(f"Error parsing question {i}: {e} - skipping this question.")
                skipped_count += 1
                continue
            questions.append(question)

        logger.info(f"Loaded {len(questions)} benchmark questions from {benchmark_qs_path}")
        if self.config_benchmark.reverse_order:
            logger.info("Reversing order of benchmark questions.")
            questions = questions[::-1]
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} questions due to parsing errors.")
        if self.config_benchmark.limit is not None:
            questions = questions[: self.config_benchmark.limit]
            logger.info(f"Limited to {len(questions)} benchmark questions for debugging.")
        elif self.config_benchmark.question_ids:
            question_ids = [str(q) for q in self.config_benchmark.question_ids]  # ids need to be str
            questions = [q for q in questions if q.id in question_ids]
            logger.info(f"Filtered to {len(questions)} benchmark questions based on question IDs {question_ids}")
        return questions

    def format_benchmark_questions(self, questions: list[BenchmarkQuestion]) -> list[BenchmarkQuestion]:
        """
        Prepare benchmark questions for the experiment.
        """
        multi_wsi_questions_count = 0
        single_wsi_questions_count = 0
        for question in tqdm(questions, desc="Formatting benchmark questions", total=len(questions)):
            match question.data_type:
                case DataType.MULTIPLE_WSI:
                    multi_wsi_questions_count += 1
                case DataType.SINGLE_WSI:
                    single_wsi_questions_count += 1
            question.format_question(
                root_path_to_dataset=self.root_path_to_dataset,
                root_working_dir=self.working_dir,
                extra_instructions=self.config_baseline.question_instructions,
            )
        logger.info(
            f"Formatted {multi_wsi_questions_count} multiple WSI questions and {single_wsi_questions_count} single WSI questions."
        )
        return questions

    def read_question_ground_truth(self, question: BenchmarkQuestion) -> pd.DataFrame:
        gt_path = self.benchmark_category_path / "ground_truth" / f"question_{question.id}" / "answer.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file {gt_path} does not exist for question ID {question.id}")
        # copy the answer to the working dir of the question for easy access
        q_working_dir = question.get_question_working_dir(self.working_dir)
        _ = shutil.copy(gt_path, q_working_dir / "answer_gt.json")
        try:
            return pd.read_json(gt_path)
        except ValueError as e:
            raise ValueError(f"Invalid JSON in ground truth file {gt_path} for question ID {question.id}") from e

    def prepare_benchmark_questions(self) -> list[BenchmarkQuestion]:
        """
        Prepare the experiment by reading and formatting benchmark questions.
        """
        questions = self.read_benchmark_questions()
        formatted_questions = self.format_benchmark_questions(questions)
        logger.info(f"Prepared {len(formatted_questions)} benchmark questions for the experiment.")
        return formatted_questions

    def _log_eval_results(self, eval_results: pd.DataFrame, data_type: DataType | None) -> None:
        n_total_questions = len(eval_results)
        n_successful_questions = int(eval_results[EvalKey.SUCCESS].sum())
        n_failed_questions = n_total_questions - n_successful_questions
        n_rate_limit_errors = 0
        n_eval_errors = 0
        n_other_failures = n_failed_questions

        # Calculate task completion rate
        n_completed_questions = int(eval_results[EvalKey.COMPLETION].sum())
        completion_rate = eval_results[EvalKey.COMPLETION].mean()

        if EvalKey.ERROR in eval_results.columns:
            logger.warning(f"Failed questions: {n_failed_questions}")
            # Count rate limit and eval errors specifically
            n_rate_limit_errors = int(
                eval_results[EvalKey.ERROR].str.contains(f"{RATE_LIMIT_ERROR_CODE}", na=False).sum()
            )
            n_eval_errors = int(eval_results[EvalKey.ERROR].str.contains(f"{EVALUATION_ERROR_MSG}", na=False).sum())
            n_other_failures = n_other_failures - n_rate_limit_errors - n_eval_errors

        # Standard metrics
        success_rate = eval_results[EvalKey.SUCCESS].mean()
        avg_score_successful = (
            eval_results.loc[eval_results[EvalKey.SUCCESS], EvalKey.SCORE].mean()
            if eval_results[EvalKey.SUCCESS].any()
            else 0.0
        )
        avg_score_overall = eval_results[EvalKey.SCORE].mean()

        # Adjusted metrics (excluding rate limit errors)
        n_evaluable_questions = n_total_questions - n_rate_limit_errors
        adjusted_success_rate = n_successful_questions / n_evaluable_questions if n_evaluable_questions > 0 else 0.0
        adjusted_completion_rate = n_completed_questions / n_evaluable_questions if n_evaluable_questions > 0 else 0.0
        total_score = eval_results[EvalKey.SCORE].sum()
        adjusted_avg_score = total_score / n_evaluable_questions if n_evaluable_questions > 0 else 0.0

        # Rate limit impact
        rate_limit_impact = n_rate_limit_errors / n_total_questions if n_total_questions > 0 else 0.0

        if data_type is not None:
            logger.info("*" * 10 + f" {data_type.value} " + "*" * 10)
        logger.info(f"Task completion rate                       : {completion_rate:.2%}")
        logger.info(f"Task success rate                          : {success_rate:.2%}")
        logger.info(f"Average score for successful questions     : {avg_score_successful:.2%}")
        logger.info(f"Average overall score (including failed)   : {avg_score_overall:.2%}")
        logger.info(f"Adjusted completion rate (excl rate limits): {adjusted_completion_rate:.2%}")
        logger.info(f"Adjusted success rate (excl rate limits)   : {adjusted_success_rate:.2%}")
        logger.info(f"Adjusted avg score (excl rate limits)      : {adjusted_avg_score:.2%}")
        logger.info(f"Rate limit impact                          : {rate_limit_impact:.2%}")
        logger.info(f"Total errors                               : {n_failed_questions}")
        logger.info(f"Eval errors                                : {n_eval_errors}")
        logger.info(f"Other errors                               : {n_other_failures}")
        logger.info(f"Rate limit errors (429)                    : {n_rate_limit_errors}")

        prefix = f"{self.benchmark_category}/" if data_type is None else f"{self.benchmark_category}/{data_type}/"
        if self.aml_job_id is not None:
            mlflow.log_metrics(
                {
                    f"{prefix}completion_rate": float(completion_rate),
                    f"{prefix}success_rate": float(success_rate),
                    f"{prefix}avg_score_successful": float(avg_score_successful),
                    f"{prefix}avg_score_overall": float(avg_score_overall),
                    f"{prefix}adjusted_completion_rate": float(adjusted_completion_rate),
                    f"{prefix}adjusted_success_rate": float(adjusted_success_rate),
                    f"{prefix}adjusted_avg_score": float(adjusted_avg_score),  # This is comparable across runs
                    f"{prefix}rate_limit_impact": float(rate_limit_impact),
                    f"{prefix}n_eval_errors": n_eval_errors,
                    f"{prefix}n_rate_limit_errors": n_rate_limit_errors,
                    f"{prefix}n_other_errors": n_other_failures,
                    f"{prefix}n_questions": n_total_questions,
                    f"{prefix}n_evaluable_questions": n_evaluable_questions,
                    f"{prefix}n_completed_questions": n_completed_questions,
                    f"{prefix}n_failed_questions": n_failed_questions,
                    f"{prefix}n_successful_questions": n_successful_questions,
                }
            )

    def setup_huggingface_cache(self) -> None:
        if self.config.run.huggingface_cache is not None:
            # symnlink to the default huggingface cache
            remote_huggingface_cache = Path(self.config.run.huggingface_cache)
            logger.info("Creating symlink to remote huggingface cache...")
            if not remote_huggingface_cache.exists():
                raise FileNotFoundError(
                    f"Remote Hugging Face cache directory {remote_huggingface_cache} does not exist."
                )
            symlink_path = Path("~/.cache/huggingface")
            symlink_path.parent.mkdir(exist_ok=True, parents=True)
            try:
                symlink_path.symlink_to(remote_huggingface_cache)
                logger.info(f"Successfully created symlink {symlink_path} -> {remote_huggingface_cache}")
            except Exception as e:
                logger.error(f"Error setting up Hugging Face cache symlink: {e}")

    def run(self) -> None:
        save_config(self.config, self.root_working_dir / "config.yaml")
        self.setup_huggingface_cache()
        logger.info(f"Running experiment: {self.experiment_description}")
        logger.info(f"Root working directory: {self.root_working_dir}")
        agent = self.get_agent()
        questions = self.prepare_benchmark_questions()
        eval_results_list = []

        for question in tqdm(questions, desc=f"Running experiment {self.experiment_description}", total=len(questions)):
            # 1. Run the model
            answer_dict = {}
            run_success = True
            eval_result = pd.Series(dtype=object)
            try:
                logger.info(f"{'*' * 10} Running question ID {question.id} {'*' * 10}")
                answer_dict = agent.run(question.question)
                answer_path = question.get_question_working_dir(self.working_dir) / "answer.json"
                if not answer_path.exists():
                    error_msg = "Answer file not saved by the agent."
                    logger.error(f"Error: {error_msg} for question ID {question.id}")
                    run_success = False
                    eval_result = pd.Series(
                        {
                            EvalKey.ID: question.id,
                            EvalKey.SUCCESS: False,
                            EvalKey.COMPLETION: False,
                            EvalKey.SCORE: 0.0,
                            EvalKey.ERROR: error_msg,
                            EvalKey.DATA_TYPE: question.data_type,
                        }
                    )
            except Exception as e:
                logger.error(f"Error running experiment for question {question.id}: {e}")
                answer_dict = {BaselineOutputColumns.ERROR: str(e)}
                run_success = False
                eval_result = pd.Series(
                    {
                        EvalKey.ID: question.id,
                        EvalKey.COMPLETION: False,
                        EvalKey.SUCCESS: False,
                        EvalKey.SCORE: 0.0,
                        EvalKey.ERROR: str(e),
                        EvalKey.DATA_TYPE: question.data_type,
                    }
                )
            finally:
                question.save_question_answer(output_dir=self.output_dir, answer_dict=answer_dict)

            # 2. Evaluate if it's the correct answer if success
            if run_success:
                try:
                    gt_answer = self.read_question_ground_truth(question)
                    eval_result = question.evaluate_answer(
                        output_dir=self.output_dir,
                        working_dir=self.working_dir,
                        ground_truth=gt_answer,
                    )
                except Exception as e:
                    err_message = f"{EVALUATION_ERROR_MSG} {question.id}: {e}"
                    logger.error(err_message)
                    eval_result = pd.Series(
                        {
                            EvalKey.ID: question.id,
                            EvalKey.COMPLETION: True,
                            EvalKey.SUCCESS: False,
                            EvalKey.SCORE: 0.0,
                            EvalKey.ERROR: err_message,
                            EvalKey.DATA_TYPE: question.data_type,
                        }
                    )

            eval_results_list.append(eval_result)
            if self.aml_job_id is not None:
                question_id_int = int(question.id)
                mlflow.log_metrics(
                    {
                        f"{self.benchmark_category}/q_completion": int(eval_result[EvalKey.COMPLETION]),
                        f"{self.benchmark_category}/q_success": int(eval_result[EvalKey.SUCCESS]),
                        f"{self.benchmark_category}/q_score": float(eval_result[EvalKey.SCORE]),
                    },
                    step=question_id_int,
                )

            if self.config.run.sleep_sec != 0:
                logger.info(f"Sleeping for {self.config.run.sleep_sec} seconds to avoid rate limits...")
                time.sleep(self.config.run.sleep_sec)

        eval_results = pd.DataFrame(eval_results_list)
        eval_results.to_json(
            self.output_dir / "eval_results_all_questions.json",
            orient='records',
            indent=4,
        )
        logger.info(f"Results saved to {self.output_dir}")

        self._log_eval_results(eval_results=eval_results, data_type=None)
        for data_type in DataType:
            data_type_results = eval_results[eval_results[EvalKey.DATA_TYPE] == data_type.value]
            if not data_type_results.empty:
                self._log_eval_results(eval_results=data_type_results, data_type=data_type)

        if self.aml_job_id is not None:
            mlflow.end_run()
