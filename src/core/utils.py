"""
Utility: safe JSON parsing for LLM outputs.
Place this file at: src/core/utils.py  (or add the function to an existing utils module)
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


def safe_json_parse(
        text: str,
        fallback: Optional[Dict] = None
) -> Tuple[Dict[str, Any], bool, str]:
    """
    Robustly parse JSON from LLM output.

    Handles:
      - Clean JSON
      - JSON wrapped in ```json ... ``` markdown blocks
      - JSON buried in surrounding prose
      - Qwen3's <think>...</think> preamble before JSON

    Returns:
        (parsed_dict, success_bool, error_message)
    """
    if not text or not text.strip():
        return fallback or {}, False, "Empty input"

    cleaned = text.strip()

    # Step 0: Strip Qwen3 <think>...</think> blocks (common with qwen3 /think mode)
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', cleaned).strip()

    # Step 1: Try direct parse
    try:
        return json.loads(cleaned), True, ""
    except json.JSONDecodeError:
        pass

    # Step 2: Extract from ```json ... ``` code blocks
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', cleaned)
    if match:
        try:
            return json.loads(match.group(1).strip()), True, ""
        except json.JSONDecodeError:
            pass

    # Step 3: Find outermost { ... }
    brace_start = cleaned.find('{')
    if brace_start != -1:
        # Find matching closing brace (handle nesting)
        depth = 0
        for i in range(brace_start, len(cleaned)):
            if cleaned[i] == '{':
                depth += 1
            elif cleaned[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[brace_start:i + 1]
                    try:
                        return json.loads(candidate), True, ""
                    except json.JSONDecodeError:
                        break

    # Step 4: Last resort — try to find any JSON array
    bracket_start = cleaned.find('[')
    if bracket_start != -1:
        depth = 0
        for i in range(bracket_start, len(cleaned)):
            if cleaned[i] == '[':
                depth += 1
            elif cleaned[i] == ']':
                depth -= 1
                if depth == 0:
                    candidate = cleaned[bracket_start:i + 1]
                    try:
                        return json.loads(candidate), True, ""
                    except json.JSONDecodeError:
                        break

    return fallback or {}, False, f"Failed to parse JSON from: {text[:200]}"


# Quick self-test
# if __name__ == "__main__":
#     tests = [
#         # Clean JSON
#         ('{"a": 1}', True),
#         # Markdown wrapped
#         ('Here is the plan:\n```json\n{"steps": [1,2]}\n```\nDone.', True),
#         # Prose + JSON
#         ('I think the answer is {"result": "ok"} and that is final.', True),
#         # Qwen3 think block
#         ('<think>\nLet me analyze...\n</think>\n{"plan": "done"}', True),
#         # Empty
#         ('', False),
#         # Garbage
#         ('This is not JSON at all', False),
#         # Nested braces
#         ('{"a": {"b": {"c": 1}}, "d": 2}', True),
#     ]
#
#     for text, expected_success in tests:
#         result, success, error = safe_json_parse(text)
#         status = "✓" if success == expected_success else "✗"
#         print(f"  {status} success={success} expected={expected_success} | {text[:50]}")
#         if not success and error:
#             print(f"    error: {error}")