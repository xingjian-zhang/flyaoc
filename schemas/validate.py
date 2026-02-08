#!/usr/bin/env python3
"""Standalone CLI tool for JSON schema validation.

This is a command-line utility for validating agent output files against the
benchmark schema. It is independent from the core validation module used by
agents (agent/core/validation.py).

Usage:
    python validate.py output.json
    python validate.py --batch outputs_dir/
"""

import argparse
import json
import re
import sys
from pathlib import Path

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
    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message

    def __str__(self):
        return f"{self.path}: {self.message}" if self.path else self.message


def validate_pattern(value: str, pattern_name: str, path: str) -> list[ValidationError]:
    if value is None:
        return []
    pattern = PATTERNS.get(pattern_name)
    if pattern and not pattern.match(value):
        return [ValidationError(path, f"Invalid format: '{value}'")]
    return []


def validate_enum(value: str, allowed: set, path: str) -> list[ValidationError]:
    if value not in allowed:
        return [ValidationError(path, f"Invalid value '{value}'")]
    return []


def validate_evidence(evidence: dict, path: str) -> list[ValidationError]:
    errors = []
    if not isinstance(evidence, dict):
        return [ValidationError(path, "Must be an object")]

    if "pmcid" in evidence:
        errors.extend(validate_pattern(evidence["pmcid"], "pmcid", f"{path}.pmcid"))

    if "text" in evidence and len(evidence["text"]) > 500:
        errors.append(ValidationError(f"{path}.text", "Exceeds 500 characters"))

    return errors


def validate_go_annotation(ann: dict, idx: int) -> list[ValidationError]:
    path = f"task1_function[{idx}]"
    errors = []

    if "go_id" not in ann:
        errors.append(ValidationError(path, "Missing 'go_id'"))
    else:
        errors.extend(validate_pattern(ann["go_id"], "go_id", f"{path}.go_id"))

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


def validate_file(filepath: Path) -> tuple[bool, list[ValidationError]]:
    try:
        with open(filepath) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [ValidationError("", f"Invalid JSON: {e}")]
    except Exception as e:
        return False, [ValidationError("", f"Error reading file: {e}")]

    errors = validate_output(data)
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate agent output")
    parser.add_argument("path", type=Path, help="JSON file or directory")
    parser.add_argument("--batch", action="store_true", help="Validate all JSON files in directory")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show errors")
    args = parser.parse_args()

    if args.batch or args.path.is_dir():
        files = list(args.path.glob("*.json"))
        if not files:
            print(f"No JSON files found in {args.path}")
            sys.exit(1)

        total_valid = 0
        for filepath in sorted(files):
            valid, errors = validate_file(filepath)
            if valid:
                total_valid += 1
                if not args.quiet:
                    print(f"OK: {filepath.name}")
            else:
                print(f"INVALID: {filepath.name}")
                for err in errors:
                    print(f"  {err}")

        print(f"\n{total_valid}/{len(files)} valid")
        sys.exit(0 if total_valid == len(files) else 1)

    else:
        valid, errors = validate_file(args.path)
        if valid:
            print(f"OK: {args.path}")
            sys.exit(0)
        else:
            print(f"INVALID: {args.path}")
            for err in errors:
                print(f"  {err}")
            sys.exit(1)


if __name__ == "__main__":
    main()
