import json
import re
from typing import Any

JSON_START_TAG = "JSON_START"
JSON_END_TAG = "JSON_END"


def extract_json_from_text(
    text: str, json_start_tag: str = JSON_START_TAG, json_end_tag: str = JSON_END_TAG
) -> Any | None:
    """Extract JSON from the LLM's output."""
    pattern = rf"{json_start_tag}(.*?){json_end_tag}"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON block between '{json_start_tag}' and '{json_end_tag}' found in the output: {text}")
    if len(matches) > 1:
        raise ValueError(f"Multiple JSON blocks found in the output {text}")
    json_content = matches[0].strip()
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON found in the output {text} \n {e}")


def extract_code_from_text(text: str) -> str | None:
    """Extract code from the LLM's output."""
    pattern = r"<code>(.*?)</code>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None
