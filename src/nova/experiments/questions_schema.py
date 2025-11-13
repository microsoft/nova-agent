import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from pyparsing import Path

from nova.baselines.base import BaselineOutputColumns, BaselineOutputType
from nova.evaluation.evaluator import EvalKey, evaluate_answer

logger = logging.getLogger(__name__)


class DataType(StrEnum):
    SUMMARY = "summary"
    SINGLE_WSI = "single_wsi"
    MULTIPLE_WSI = "multiple_wsi"


@dataclass
class BenchmarkQuestion:
    """
    Schema for a benchmark question in the experiment.
    """

    id: str
    data_type: DataType
    question: str
    id_column: str | None
    slide_relative_path: str | None = None
    dataset_relative_path: str | None = None
    path_to_metadata: str | None = None
    processing_dir_relative_path: str | None = None
    additional_instructions: str = ""
    output_instructions: str = ""
    rationale: str = ""
    columns_to_compare_and_tolerance: dict[str, float | None] = field(default_factory=dict)
    is_pathologist_verified: bool = False
    is_biomedical_scientist_verified: bool = False

    @property
    def eval_result_filename(self) -> str:
        return "evaluation_result.json"

    @property
    def detailed_eval_result_filename(self) -> str:
        return "evaluation_result_detailed.json"

    def get_question_output_dir(self, output_dir: Path) -> Path:
        output_dir = output_dir / f"question_{self.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_question_working_dir(self, root_working_dir: Path) -> Path:
        output_dir = root_working_dir / f"question_{self.id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def add_special_instructions_to_question(self, root_working_dir: Path, root_path_to_dataset: Path) -> None:
        q_working_dir = self.get_question_working_dir(root_working_dir)
        fmt_kwargs = {"working_dir": q_working_dir}

        # add dataset_relative_path
        if self.dataset_relative_path:
            path_to_dataset = root_path_to_dataset / self.dataset_relative_path
            fmt_kwargs["path_to_dataset"] = path_to_dataset

        # add metadata path if available
        if self.path_to_metadata:
            path_to_metadata = root_path_to_dataset / self.path_to_metadata

            if not path_to_metadata.exists():
                raise FileNotFoundError(f"Metadata path {path_to_metadata} does not exist for question ID {self.id}")

            fmt_kwargs["path_to_metadata"] = path_to_metadata

        # add processing data path if available
        if self.processing_dir_relative_path:
            path_to_processing_dir = root_path_to_dataset / self.processing_dir_relative_path

            if not path_to_processing_dir.exists():
                raise FileNotFoundError(
                    f"Processing directory path {path_to_processing_dir} does not exist for question ID {self.id}"
                )

            fmt_kwargs["processing_dir_path"] = path_to_processing_dir

        if self.additional_instructions:
            try:
                self.question += f" Additional instructions: {self.additional_instructions.format(**fmt_kwargs)}."
            except KeyError as e:
                logger.warning(
                    f"Failed to format additional instructions for Q {self.id} {self.additional_instructions=}: {e}"
                )

        if self.output_instructions:
            self.question += f" Output instructions: {self.output_instructions}."

    def format_multiple_wsi_question(self, root_path_to_dataset: Path) -> None:
        """Format a multiple WSI question with validation."""
        if self.dataset_relative_path is None:
            raise ValueError(
                f"dataset_relative_path is required for multiple WSI questions, but got None for ID {self.id}"
            )

        path_to_dataset = root_path_to_dataset / self.dataset_relative_path
        if not path_to_dataset.exists():
            raise FileNotFoundError(f"Dataset path {path_to_dataset} does not exist for question ID {self.id}")

        try:
            self.question = self.question.format(path_to_dataset=path_to_dataset)
        except KeyError as e:
            raise ValueError(f"Error formatting multiple wsi question {self.id}: missing key {e}") from e

    def format_single_wsi_question(self, root_path_to_dataset: Path) -> None:
        """Format a single WSI question with validation."""
        if self.slide_relative_path is None:
            raise ValueError(f"slide_relative_path is required for single WSI questions, but got None for ID {self.id}")

        path_to_slide = root_path_to_dataset / self.slide_relative_path
        if not path_to_slide.exists():
            raise FileNotFoundError(f"Slide path {path_to_slide} does not exist for question ID {self.id}")

        try:
            self.question = self.question.format(path_to_slide=path_to_slide)
        except KeyError as e:
            raise ValueError(f"Error formatting single wsi question {self.id}: missing key {e}") from e

    def format_question(self, root_path_to_dataset: Path, root_working_dir: Path, extra_instructions: str = "") -> None:
        """Format the question with validation."""
        if self.data_type == DataType.SINGLE_WSI:
            self.format_single_wsi_question(root_path_to_dataset)
        elif (
            self.data_type in [DataType.MULTIPLE_WSI, DataType.SUMMARY]
        ):  # we assume that if a question is summary type, then it must be a multiple_wsi data_type and should be formatted similarly
            self.format_multiple_wsi_question(root_path_to_dataset)
        else:
            raise ValueError(f"Unknown data type {self.data_type} for question ID {self.id}")
        self.add_special_instructions_to_question(
            root_working_dir=root_working_dir, root_path_to_dataset=root_path_to_dataset
        )
        if extra_instructions:
            self.question += f" {extra_instructions}"

    def save_question(self, question_output_dir: Path) -> None:
        question_file = question_output_dir / "question.json"
        with open(question_file, "w") as f:
            question_dict = dict(**self.__dict__)
            json.dump(question_dict, f, indent=4)

    def save_question_answer(self, output_dir: Path, answer_dict: BaselineOutputType) -> None:
        """
        Save the answer to the question.
        """
        question_output_dir = self.get_question_output_dir(output_dir)
        self.save_question(question_output_dir)
        for key, value in answer_dict.items():
            extension = "json" if "json" in key.lower() else "txt"
            output_file = question_output_dir / f"{key.replace('_json', '')}.{extension}"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                if extension == "json":
                    try:
                        json.dump(json.loads(value) if isinstance(value, str) else value, f, indent=4)
                    except Exception as e:
                        logger.warning(
                            f"Failed to write JSON for key {key} in question {self.id}: {e}. Writing as string."
                        )
                        with open(output_file.with_suffix(".txt"), "w") as txt_f:
                            txt_f.write(str(value))
                else:
                    f.write(str(value))
        logger.info(f"Question and outputs ID {self.id} saved to: {question_output_dir}")

    def evaluate_answer(self, output_dir: Path, working_dir: Path, ground_truth: pd.DataFrame) -> pd.Series:
        """
        Evaluate the answer against the ground truth.
        """
        question_output_dir = self.get_question_output_dir(output_dir)
        question_working_dir = self.get_question_working_dir(working_dir)
        agent_answer_path = question_working_dir / BaselineOutputColumns.ANSWER_JSON.replace("_", ".")

        eval_result, combined_answers = evaluate_answer(
            agent_answer_path=agent_answer_path,
            correct_answer=ground_truth,
            id_column=self.id_column,
            columns_to_compare_and_tolerance=self.columns_to_compare_and_tolerance,
        )

        eval_result = pd.concat([pd.Series({EvalKey.ID: self.id}), eval_result])  # Add to the front
        eval_result[EvalKey.DATA_TYPE] = self.data_type
        eval_result[EvalKey.COMPLETION] = True  # if it got to evaluation, then the task was completed

        eval_result.to_frame().T.to_json(
            question_output_dir / self.eval_result_filename,
            orient='records',
            indent=4,
        )
        if combined_answers is not None:
            combined_answers.to_json(
                question_output_dir / self.detailed_eval_result_filename,
                orient='records',
                indent=4,
            )

        return eval_result
