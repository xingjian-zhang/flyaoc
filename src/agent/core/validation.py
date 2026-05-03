"""Core validation logic for agent outputs.

This module provides the reusable validation functions used by agent tools
(LangChain, MCP) to validate annotation outputs. It contains:
- Regex patterns for ID formats (FBgn, GO, FBbt, FBdv, PMC)
- Enum validation for qualifiers, aspects, expression types
- Schema validation for GO annotations, expression records, synonyms
- The submit_annotations_core() function used by tool wrappers
"""

import re

# Validation patterns
PATTERNS = {
    "gene_id": re.compile(r"^FBgn\d{7}$"),
    "go_id": re.compile(r"^GO:\d{7}$"),
    "fbbt_id": re.compile(r"^FBbt:\d{8}$"),
    "fbdv_id": re.compile(r"^FBdv:\d{8}$"),
    "pmcid": re.compile(r"^PMC\d+$"),
}

# Allowed enum values
QUALIFIERS = {
    "located_in",
    "part_of",
    "enables",
    "contributes_to",
    "involved_in",
    "acts_upstream_of",
    "acts_upstream_of_positive_effect",
    "acts_upstream_of_negative_effect",
    "acts_upstream_of_or_within",
    "acts_upstream_of_or_within_positive_effect",
    "acts_upstream_of_or_within_negative_effect",
    "is_active_in",
    "colocalizes_with",
}

ASPECTS = {"P", "F", "C"}
EXPRESSION_TYPES = {"polypeptide", "transcript"}


class ValidationError:
    """Represents a validation error."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message

    def __str__(self):
        return f"{self.path}: {self.message}" if self.path else self.message

    def to_dict(self):
        return {"path": self.path, "message": self.message}


def validate_pattern(value: str, pattern_name: str, path: str) -> list[ValidationError]:
    """Validate a value against a regex pattern."""
    if value is None:
        return []
    pattern = PATTERNS.get(pattern_name)
    if pattern and not pattern.match(value):
        return [ValidationError(path, f"Invalid format: '{value}'")]
    return []


def validate_enum(value: str, allowed: set, path: str) -> list[ValidationError]:
    """Validate a value against allowed enum values."""
    if value not in allowed:
        return [ValidationError(path, f"Invalid value '{value}', must be one of: {allowed}")]
    return []


def validate_evidence(evidence: dict, path: str) -> list[ValidationError]:
    """Validate an evidence object."""
    errors = []
    if not isinstance(evidence, dict):
        return [ValidationError(path, "Must be an object")]

    if "pmcid" in evidence:
        errors.extend(validate_pattern(evidence["pmcid"], "pmcid", f"{path}.pmcid"))

    if "text" in evidence and len(evidence["text"]) > 500:
        errors.append(ValidationError(f"{path}.text", "Exceeds 500 characters"))

    return errors


def validate_go_annotation(ann: dict, idx: int) -> list[ValidationError]:
    """Validate a GO annotation object.

    Accepts either go_id (structured) or description (natural language), not both.
    This supports the specificity gap benchmark where some GO terms are hidden.
    """
    path = f"task1_function[{idx}]"
    errors = []

    has_go_id = "go_id" in ann
    has_description = "description" in ann

    # Must have exactly one of go_id or description
    if has_go_id and has_description:
        errors.append(ValidationError(path, "Cannot have both 'go_id' and 'description'"))
    elif not has_go_id and not has_description:
        errors.append(ValidationError(path, "Missing 'go_id' or 'description'"))
    elif has_go_id:
        errors.extend(validate_pattern(ann["go_id"], "go_id", f"{path}.go_id"))
    else:  # has_description
        if not isinstance(ann["description"], str):
            errors.append(ValidationError(f"{path}.description", "Must be a string"))
        elif len(ann["description"]) > 100:
            errors.append(
                ValidationError(
                    f"{path}.description",
                    "Exceeds 100 characters - description should be a short noun phrase (2-6 words)",
                )
            )

    if "qualifier" not in ann:
        errors.append(ValidationError(path, "Missing 'qualifier'"))
    else:
        errors.extend(validate_enum(ann["qualifier"], QUALIFIERS, f"{path}.qualifier"))

    if "aspect" not in ann:
        errors.append(ValidationError(path, "Missing 'aspect'"))
    else:
        errors.extend(validate_enum(ann["aspect"], ASPECTS, f"{path}.aspect"))

    if "is_negated" in ann and not isinstance(ann["is_negated"], bool):
        errors.append(ValidationError(f"{path}.is_negated", "Must be boolean"))

    if "evidence" in ann:
        errors.extend(validate_evidence(ann["evidence"], f"{path}.evidence"))

    return errors


def validate_expression_record(rec: dict, idx: int) -> list[ValidationError]:
    """Validate an expression record object."""
    path = f"task2_expression[{idx}]"
    errors = []

    if "expression_type" not in rec:
        errors.append(ValidationError(path, "Missing 'expression_type'"))
    else:
        errors.extend(
            validate_enum(rec["expression_type"], EXPRESSION_TYPES, f"{path}.expression_type")
        )

    if "anatomy_id" in rec and rec["anatomy_id"]:
        errors.extend(validate_pattern(rec["anatomy_id"], "fbbt_id", f"{path}.anatomy_id"))

    if "stage_id" in rec and rec["stage_id"]:
        errors.extend(validate_pattern(rec["stage_id"], "fbdv_id", f"{path}.stage_id"))

    if "evidence" in rec:
        errors.extend(validate_evidence(rec["evidence"], f"{path}.evidence"))

    return errors


def validate_synonyms(synonyms: dict) -> list[ValidationError]:
    """Validate the synonyms object."""
    path = "task3_synonyms"
    errors = []

    if not isinstance(synonyms, dict):
        return [ValidationError(path, "Must be an object")]

    for field in ["fullname_synonyms", "symbol_synonyms"]:
        if field not in synonyms:
            errors.append(ValidationError(path, f"Missing '{field}'"))
        elif not isinstance(synonyms[field], list):
            errors.append(ValidationError(f"{path}.{field}", "Must be an array"))
        elif not all(isinstance(s, str) for s in synonyms[field]):
            errors.append(ValidationError(f"{path}.{field}", "All items must be strings"))

    return errors


def validate_output(data: dict) -> list[ValidationError]:
    """Validate a complete agent output against the schema."""
    errors = []

    # Required fields
    if "gene_id" not in data:
        errors.append(ValidationError("", "Missing 'gene_id'"))
    else:
        errors.extend(validate_pattern(data["gene_id"], "gene_id", "gene_id"))

    if "gene_symbol" not in data:
        errors.append(ValidationError("", "Missing 'gene_symbol'"))

    # Task 1
    if "task1_function" not in data:
        errors.append(ValidationError("", "Missing 'task1_function'"))
    elif not isinstance(data["task1_function"], list):
        errors.append(ValidationError("task1_function", "Must be an array"))
    else:
        for i, ann in enumerate(data["task1_function"]):
            errors.extend(validate_go_annotation(ann, i))

    # Task 2
    if "task2_expression" not in data:
        errors.append(ValidationError("", "Missing 'task2_expression'"))
    elif not isinstance(data["task2_expression"], list):
        errors.append(ValidationError("task2_expression", "Must be an array"))
    else:
        for i, rec in enumerate(data["task2_expression"]):
            errors.extend(validate_expression_record(rec, i))

    # Task 3
    if "task3_synonyms" not in data:
        errors.append(ValidationError("", "Missing 'task3_synonyms'"))
    else:
        errors.extend(validate_synonyms(data["task3_synonyms"]))

    return errors


def submit_annotations_core(annotations: dict) -> dict:
    """Submit final gene annotations for validation.

    This is the framework-agnostic core function.

    Args:
        annotations: Complete output dictionary containing gene_id, gene_symbol,
                    task1_function, task2_expression, task3_synonyms

    Returns:
        Dictionary with valid (bool), errors (list), and annotation_count (dict)
    """
    errors = validate_output(annotations)

    result = {
        "valid": len(errors) == 0,
        "errors": [str(e) for e in errors],
        "annotation_count": {},
    }

    # Add annotation counts if structure is valid enough
    if isinstance(annotations.get("task1_function"), list):
        result["annotation_count"]["task1_go_terms"] = len(annotations["task1_function"])

    if isinstance(annotations.get("task2_expression"), list):
        result["annotation_count"]["task2_expression"] = len(annotations["task2_expression"])

    if isinstance(annotations.get("task3_synonyms"), dict):
        syns = annotations["task3_synonyms"]
        fullname_count = len(syns.get("fullname_synonyms", []))
        symbol_count = len(syns.get("symbol_synonyms", []))
        result["annotation_count"]["task3_synonyms"] = fullname_count + symbol_count

    return result
