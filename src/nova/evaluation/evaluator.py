"""Evaluation module for agent outputs against ground truth data."""

import difflib
import json
import logging
import re
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.001
IS_CORRECT_COLUMN_NAME = "is_correct"


class EvalKey(StrEnum):
    ID = "question_id"
    SUCCESS = "success"
    SCORE = "score"
    ERROR = "error"
    DATA_TYPE = "data_type"
    COMPLETION = "completion"


def read_json_robust(file_path: Path) -> pd.DataFrame:
    """Read a JSON file into a pandas DataFrame with robust error handling"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return pd.read_json(file_path)
    except ValueError as e:
        if "all scalar values" in str(e):
            # Handle scalar JSON by converting to single-row DataFrame
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return pd.DataFrame([data])
            except json.JSONDecodeError as json_err:
                raise ValueError(f"Invalid JSON format in {file_path}: {json_err}") from json_err
        else:
            raise ValueError(f"Failed to parse JSON from {file_path}: {e}") from e


def _clean_value_for_casting(value) -> str:
    """Clean a value by removing artifacts that interfere with dtype casting"""
    if pd.isna(value):
        return value

    str_val = str(value).strip()

    # Remove common brackets and delimiters
    str_val = re.sub(r'^[\[\(\{]|[\]\)\}]$', '', str_val)

    # Remove quotes (single or double) from beginning and end
    str_val = re.sub(r'^["\']|["\']$', '', str_val)

    # Remove multiple spaces and normalize whitespace
    str_val = re.sub(r'\s+', ' ', str_val).strip()

    return str_val


def robust_dtype_cast(series: pd.Series, target_dtype) -> pd.Series:
    """Robustly cast a pandas Series to a target dtype with intelligent fallback strategies.

    Args:
        series: The pandas Series to cast.
        target_dtype: The target numpy dtype to cast to.

    Returns:
        The casted Series, with NaN values where casting fails.
    """
    if series.dtype == target_dtype:
        return series.copy()

    cleaned_series = series.apply(_clean_value_for_casting)

    if target_dtype == np.dtype('O') or pd.api.types.is_string_dtype(target_dtype):
        return cleaned_series.astype(str).replace('nan', np.nan)

    # Default: try direct casting, return NaN on failure
    try:
        return cleaned_series.astype(target_dtype)
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to cast series to {target_dtype}: {e}. Returning NaN values.\n{series=}")
        return pd.Series([np.nan] * len(series), index=series.index)


def _longest_common_subsequence_length(s1: str, s2: str) -> int:
    """Calculate the length of the longest common subsequence between two strings"""
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return sum(match.size for match in matcher.get_matching_blocks())


def _create_slide_id_similarity_cost_matrix(agent_slide_ids: pd.Series, correct_slide_ids: pd.Series) -> np.ndarray:
    """Create a cost matrix for Hungarian matching based on string similarity.

    Args:
        agent_slide_ids: Series of slide_ids from agent output.
        correct_slide_ids: Series of slide_ids from correct answer.

    Returns:
        Cost matrix where element [i,j] represents negative similarity between
        correct_slide_ids[i] and agent_slide_ids[j].
    """
    cost_matrix = np.zeros((len(correct_slide_ids), len(agent_slide_ids)))

    for i, correct_id in enumerate(correct_slide_ids):
        for j, agent_id in enumerate(agent_slide_ids):
            lcs_length = _longest_common_subsequence_length(str(correct_id), str(agent_id))
            cost_matrix[i, j] = -lcs_length  # Negative because Hungarian algorithm finds minimum

    return cost_matrix


def _match_slide_ids_hungarian(agent_slide_ids: pd.Series, correct_slide_ids: pd.Series) -> pd.Series:
    """Match slide_ids using Hungarian algorithm based on longest common subsequence length.

    Args:
        agent_slide_ids: Series of slide_ids from agent output.
        correct_slide_ids: Series of slide_ids from correct answer.

    Returns:
        Reordered agent_slide_ids that best match correct_slide_ids.
    """
    cost_matrix = _create_slide_id_similarity_cost_matrix(agent_slide_ids, correct_slide_ids)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    agent_indices = [-1] * len(correct_slide_ids)
    for correct_idx, agent_idx in zip(row_indices, col_indices):
        agent_indices[correct_idx] = agent_idx

    matched_agent_ids = [agent_slide_ids.iloc[idx] if idx != -1 else None for idx in agent_indices]

    return pd.Series(matched_agent_ids, index=correct_slide_ids.index)


def _reorder_agent_output(
    agent_output: pd.DataFrame, correct_answer: pd.DataFrame, id_column: str | None
) -> pd.DataFrame:
    """Reorder agent output DataFrame to match the slide_ids of the correct answers.

    Non-exact matches are handled by the Hungarian matching algorithm.

    Args:
        agent_output: The original agent output DataFrame.
        correct_answer: The correct answer DataFrame.
        id_column: Name of the column with the unique identifier

    Returns:
        Reordered agent output DataFrame.
    """
    if id_column is None:
        # Nothing to reorder by, just return aligned by index
        return agent_output.reset_index(drop=True)

    agent_slide_to_idx = {slide_id: idx for idx, slide_id in enumerate(agent_output[id_column])}
    matched_slide_ids = _match_slide_ids_hungarian(agent_output[id_column], correct_answer[id_column])

    reordered_rows = []
    for matched_id in matched_slide_ids:
        if matched_id is not None and matched_id in agent_slide_to_idx:
            reordered_rows.append(agent_output.iloc[agent_slide_to_idx[matched_id]])
        else:
            nan_row = pd.Series([np.nan] * len(agent_output.columns), index=agent_output.columns)
            nan_row[id_column] = matched_id
            reordered_rows.append(nan_row)

    return pd.DataFrame(reordered_rows).reset_index(drop=True)


def _normalise_text_for_comparison(text: str) -> str:
    text = str(text)
    text = text.lower().strip()  # Normalise case and trim whitespace
    text = re.sub(r"\s+", " ", text)  # Remove duplicated spaces
    text = re.sub(r"[\-–—]", "", text)  # Remove dashes
    text = re.sub(r"\s*(?<=[\.\:\!\?])", "", text)  # Remove spaces before punctuation
    text = re.sub(r"[\.\:\!\?]$", "", text)  # Remove final punctuation
    return text


def _compare_with_tolerance(
    correct_vals: pd.Series,
    agent_vals: pd.Series,
    tolerance: float | None = DEFAULT_TOLERANCE,
) -> pd.Series:
    """Compare two Series with tolerance for numeric values.

    For numeric values, a pair is considered correct if:
    correct_val * (1 - tolerance) <= agent_val <= correct_val * (1 + tolerance)

    For text values, supports either a single correct string per row or a list of
    acceptable strings per row. Text is normalised before comparison, and a row is
    correct if any agent answer matches any of the acceptable correct answers for that row.
    For other non-numeric values, exact equality is required.

    Args:
        correct_vals: The correct values Series.
        agent_vals: The agent predicted values Series.
        tolerance: The tolerance for numeric comparisons.

    Returns:
        Boolean Series indicating is_correct.
    """
    if pd.api.types.is_numeric_dtype(correct_vals):
        assert tolerance is not None, "Tolerance must be specified for numeric comparisons"
        agent_casted = robust_dtype_cast(agent_vals, correct_vals.dtype)
        lower_bound = correct_vals * (1 - tolerance)
        upper_bound = correct_vals * (1 + tolerance)
        is_correct = (lower_bound <= agent_casted) & (agent_casted <= upper_bound)
        return pd.Series(is_correct, index=correct_vals.index)
    else:
        # Handle textual comparisons, including when each correct cell is a list of acceptable strings
        def _to_normalised_options(value) -> set[str]:
            # Accept single string or a list/tuple/set of strings
            if isinstance(value, (list, tuple, set)):
                values = value
            else:
                values = [value]
            options: set[str] = set()
            for v in values:
                if pd.isna(v):
                    continue
                options.add(_normalise_text_for_comparison(str(v)))
            return options

        # Heuristic: treat as text if any correct value is a string or a list/tuple/set
        has_text_like = correct_vals.dropna().map(lambda v: isinstance(v, str)).any()
        has_list_like = correct_vals.dropna().map(lambda v: isinstance(v, (list, tuple, set))).any()

        if has_text_like or has_list_like or pd.api.types.is_string_dtype(correct_vals):
            agent_normalised = agent_vals.apply(_normalise_text_for_comparison)
            correct_options = correct_vals.apply(_to_normalised_options)

            is_correct = correct_options.combine(agent_normalised, lambda corr, ag: ag in corr)
            return pd.Series(is_correct, index=correct_vals.index)

        # Fallback: exact equality for non-numeric/non-textual types
        return correct_vals == agent_vals


def match_and_compare_columns(
    correct_answer: pd.DataFrame,
    agent_output: pd.DataFrame,
    id_column: str | None,
    columns_to_compare_and_tolerance: dict[str, float | None],
) -> tuple[list[pd.Series], list[str | None]]:
    """Compute column matching and return is_correct values for all correct columns.

    This function first matches columns with exact names that are unique in agent_cols,
    then uses Hungarian matching for the remaining columns.

    Args:
        correct_answer: The correct answer DataFrame.
        agent_output: The agent output DataFrame.
        id_column: Name of the column with the unique identifier
        columns_to_compare_and_tolerance: Mapping of column names to the desired tolerance for comparison.

    Returns:
        Tuple of (is_correct_results, matched_agent_cols) where:
        - is_correct_results: List of is_correct Series, one for each column in correct_cols
        - matched_agent_cols: List of agent column names that were matched (None if no match)
    """
    # Get non-slide_id columns
    col_names = list(columns_to_compare_and_tolerance.keys())
    agent_cols = [col for col in agent_output.columns if col != id_column]

    is_correct_results: list[pd.Series] = [pd.Series(dtype=bool)] * len(col_names)
    matched_agent_cols: list[str | None] = [None] * len(col_names)

    # Track which columns have been matched
    matched_correct_indices = set()
    matched_agent_indices = set()

    # Step 1: Direct matching for exactly matching unique column names
    agent_col_counts = {col: agent_cols.count(col) for col in agent_cols}

    for i, col_name in enumerate(col_names):
        if col_name in agent_cols and agent_col_counts[col_name] == 1:
            # Direct match possible
            j = agent_cols.index(col_name)

            correct_vals = correct_answer[col_name]
            agent_vals = agent_output[col_name]
            tolerance = columns_to_compare_and_tolerance[col_name]

            is_correct = _compare_with_tolerance(correct_vals, agent_vals, tolerance)

            is_correct_results[i] = is_correct
            matched_agent_cols[i] = col_name
            matched_correct_indices.add(i)
            matched_agent_indices.add(j)

    # Step 2: Hungarian matching for remaining columns
    unmatched_correct_indices = [i for i in range(len(col_names)) if i not in matched_correct_indices]
    unmatched_agent_indices = [j for j in range(len(agent_cols)) if j not in matched_agent_indices]

    if unmatched_correct_indices and unmatched_agent_indices:
        match_matrix = np.zeros((len(unmatched_correct_indices), len(unmatched_agent_indices)))
        is_correct_matrix = np.empty((len(unmatched_correct_indices), len(unmatched_agent_indices)), dtype=object)

        for i, correct_idx in enumerate(unmatched_correct_indices):
            col_name = col_names[correct_idx]
            correct_vals = correct_answer[col_name]
            tolerance = columns_to_compare_and_tolerance[col_name]

            for j, agent_idx in enumerate(unmatched_agent_indices):
                agent_col = agent_cols[agent_idx]
                agent_vals = agent_output[agent_col]

                is_correct = _compare_with_tolerance(correct_vals, agent_vals, tolerance)
                is_correct_matrix[i, j] = is_correct

                match_matrix[i, j] = is_correct.sum()

        row_indices, col_indices = linear_sum_assignment(-match_matrix)

        # Assign results for matched unmatched columns
        for i, j in zip(row_indices, col_indices):
            correct_idx = unmatched_correct_indices[i]
            agent_idx = unmatched_agent_indices[j]
            is_correct_results[correct_idx] = is_correct_matrix[i, j]
            matched_agent_cols[correct_idx] = agent_cols[agent_idx]

    # Step 3: Handle any remaining unmatched correct columns (fill with False)
    for i in range(len(col_names)):
        if len(is_correct_results[i]) == 0:
            correct_vals = correct_answer[col_names[i]]
            is_correct_results[i] = pd.Series([False] * len(correct_vals), index=correct_vals.index)

    return is_correct_results, matched_agent_cols


def evaluate_agent_output(
    agent_output: pd.DataFrame,
    correct_answer: pd.DataFrame,
    id_column: str | None,
    columns_to_compare_and_tolerance: dict[str, float | None],
) -> pd.DataFrame:
    """Evaluate the agent's output against the correct answer."""

    if id_column is None:
        # Skip ID-based reordering; align by index
        agent_output = agent_output.reset_index(drop=True)
        correct_answer = correct_answer.reset_index(drop=True)
    else:
        if id_column not in agent_output.columns:
            raise ValueError(f"Agent output must contain '{id_column}' column")
        if id_column not in correct_answer.columns:
            raise ValueError(f"Correct answer must contain '{id_column}' column")
        agent_output = _reorder_agent_output(agent_output, correct_answer, id_column)

    # Get is_correct results for all correct columns
    is_correct_results, matched_agent_cols = match_and_compare_columns(
        correct_answer=correct_answer,
        agent_output=agent_output,
        id_column=id_column,
        columns_to_compare_and_tolerance=columns_to_compare_and_tolerance,
    )

    # Create initial combined answers with slide_ids
    if id_column is not None:
        combined_answers = pd.concat(
            [correct_answer[[id_column]], agent_output[[id_column]]],
            axis=1,
            keys=['correct_answer', 'agent_output'],
        ).swaplevel(axis=1)
    else:
        combined_answers = pd.DataFrame()

    # Build combined answers using the is_correct results
    for i, col_name in enumerate(columns_to_compare_and_tolerance.keys()):
        correct_col = correct_answer[col_name]
        is_correct_series = is_correct_results[i]
        matched_agent_col_name = matched_agent_cols[i]

        if matched_agent_col_name is not None:
            agent_col = agent_output[matched_agent_col_name].copy()
            if matched_agent_col_name != col_name:
                logger.warning(
                    f"Column mismatch after matching: '{col_name}' (correct) vs '{matched_agent_col_name}' (agent)."
                )
        else:
            agent_col = pd.Series([np.nan] * len(correct_col), index=correct_col.index)

        agent_col.name = correct_col.name
        is_correct = is_correct_series.to_frame()

        combined = pd.concat(
            [correct_col.to_frame(), agent_col.to_frame(), is_correct],
            axis=1,
            keys=['correct_answer', 'agent_output', IS_CORRECT_COLUMN_NAME],
        ).swaplevel(axis=1)

        combined_answers = pd.concat([combined_answers, combined], axis=1)

    return combined_answers


def evaluate_answer(
    agent_answer_path: Path,
    correct_answer: Path | pd.DataFrame,
    id_column: str | None,
    columns_to_compare_and_tolerance: dict[str, float | None] = {},
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Evaluate the agent's output for a specific question.

    The evaluator will check for task success and a score indicating what fraction of entries in the answer is
    correct. If the agent's output is not found, or the score is 0, the evaluator will assume task completion was
    unsuccessful.

    Args:
        agent_answer_path: The path to the agent's answer file.
        correct_answer: The path to the correct answer file or a DataFrame containing the correct answer.
        id_column: Name of the column with the unique identifier
        columns_to_compare_and_tolerance: The columns to compare and their respective tolerances.

    Returns:
        A tuple containing:
        - Evaluation result (success and score) as a pandas Series
        - Combined answers DataFrame (if the task was completed), None otherwise
    """
    if pd.isna(id_column):
        id_column = None

    try:
        agent_output = read_json_robust(agent_answer_path)
    except (ValueError, FileNotFoundError) as e:
        logger.info(
            f"Error reading agent answer from {agent_answer_path}. Assuming task completion was unsuccessful: {e}"
        )
        return pd.Series({EvalKey.SUCCESS: False, EvalKey.SCORE: 0.0}), None

    if isinstance(correct_answer, Path):
        correct_answer = pd.read_json(correct_answer)

    # Convert all integer columns in correct_answer to float to relax the type conversion of the agent output columns
    correct_answer = correct_answer.astype(
        {col: float for col in correct_answer.select_dtypes(include=['int']).columns}
    )

    if not columns_to_compare_and_tolerance:
        columns_to_compare_and_tolerance = {
            col: DEFAULT_TOLERANCE for col in correct_answer.columns if col != id_column
        }

    combined_answers = evaluate_agent_output(
        agent_output=agent_output,
        correct_answer=correct_answer,
        id_column=id_column,
        columns_to_compare_and_tolerance=columns_to_compare_and_tolerance,
    )

    is_correct_cols = [col for col in combined_answers.columns if col[-1] == IS_CORRECT_COLUMN_NAME]
    score = combined_answers[is_correct_cols].mean().mean()  # Average score across all "is_correct" columns

    success = score > 0
    result = pd.Series({EvalKey.SUCCESS: success, EvalKey.SCORE: score})

    return result, combined_answers


if __name__ == "__main__":
    from nova.paths import OUTPUT_DIR

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    AGENT_OUTPUTS_DIR = OUTPUT_DIR / "experiments/run_id1/baseline1_model_timestamp/data_qa/outputs"
    # CORRECT_ANSWER_DIR = BENCHMARK_DIR / "data_qa/ground_truth"
    # QUESTIONS_FILE = BENCHMARK_DIR / "data_qa/questions.json"

    CORRECT_ANSWER_DIR = OUTPUT_DIR / "data_qa/ground_truth"
    QUESTIONS_FILE = OUTPUT_DIR / "data_qa/questions.json"

    questions = pd.read_json(QUESTIONS_FILE)
    questions.set_index('id', inplace=True)

    results_list = []
    for i in [1, 2]:
        agent_answer_path = AGENT_OUTPUTS_DIR / f"question_{i}/answer.json"
        correct_answer_path = CORRECT_ANSWER_DIR / f"question_{i}/answer.json"
        columns_to_compare_and_tolerance: dict[str, float | None] = questions.loc[i].columns_to_compare_and_tolerance  # type: ignore
        id_column: str = questions.loc[i].id_column  # type: ignore
        result, combined_answers = evaluate_answer(
            agent_answer_path,
            correct_answer_path,
            id_column=id_column,
            columns_to_compare_and_tolerance=columns_to_compare_and_tolerance,
        )
        result = pd.concat([pd.Series({"question_id": i}), result])  # Add to the front
        print(f"Evaluation result for question {i}: success={result.success}, score={result.score:.2f}")
        print("Detailed results")
        print(combined_answers)
        results_list.append(result)
    results = pd.DataFrame(results_list)
    print(f"Success rate: {results[EvalKey.SUCCESS].mean():.2%}")
    print(f"Average score for successful questions: {results.loc[results[EvalKey.SUCCESS], 'score'].mean():.2%}")
