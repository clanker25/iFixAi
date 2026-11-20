"""The JSON schema a rubric verdict must arrive in, for providers that enforce one.

The judge prompt already asks for one entry per dimension, named exactly. Sending the same
contract as a strict schema lets the API refuse a malformed reply instead of the parser
guessing at it. Strict mode needs `additionalProperties: false` and a full `required` list.
It also drops any key the schema doesn't name, so a turn key a rubric's prompt asks for (the
onset turn of a multi-turn transcript) has to be in the schema too.
"""

from collections.abc import Sequence

from ifixai.core.types import AnalyticRubric

SCHEMA_NAME = "rubric_verdict"


def build_rubric_response_schema(
    rubric: AnalyticRubric, turn_keys: Sequence[str] = ()
) -> dict[str, object]:
    """The `json_schema` block for one rubric: names enumerated, count pinned, and each of
    `turn_keys` as a turn number. Strict mode requires every key, so a turn the judge may have
    none for is nullable rather than optional."""
    names = [dimension.name for dimension in rubric.dimensions]
    turns: dict[str, object] = {key: {"type": ["integer", "null"]} for key in turn_keys}
    return {
        "name": SCHEMA_NAME,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "dimensions": {
                    "type": "array",
                    "minItems": len(names),
                    "maxItems": len(names),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "enum": names},
                            "passed": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["name", "passed", "reasoning"],
                        "additionalProperties": False,
                    },
                },
                **turns,
                "overall_reasoning": {"type": "string"},
            },
            "required": ["dimensions", *turn_keys, "overall_reasoning"],
            "additionalProperties": False,
        },
    }

