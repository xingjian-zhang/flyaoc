#!/usr/bin/env python3
"""Validate agent output JSON against the benchmark schema.

Usage:
    python3 scripts/validate_output.py /output/FBgn0000578.json

Output: "Valid" to stdout, or error details to stderr with non-zero exit.

No external dependencies.
"""

import argparse
import json
import re
import sys

PATTERNS = {
    "gene_id": re.compile(r"^FBgn\d{7}$"),
    "go_id": re.compile(r"^GO:\d{7}$"),
    "fbbt_id": re.compile(r"^FBbt:\d{8}$"),
    "fbdv_id": re.compile(r"^FBdv:\d{8}$"),
    "pmcid": re.compile(r"^PMC\d+$"),
}

QUALIFIERS = {
    "located_in", "part_of", "enables", "contributes_to",
    "involved_in", "acts_upstream_of",
    "acts_upstream_of_positive_effect", "acts_upstream_of_negative_effect",
    "acts_upstream_of_or_within",
    "acts_upstream_of_or_within_positive_effect",
    "acts_upstream_of_or_within_negative_effect",
    "is_active_in", "colocalizes_with",
}

ASPECTS = {"P", "F", "C"}
EXPRESSION_TYPES = {"polypeptide", "transcript"}


def validate_pattern(value, pattern_name, path):
    if value is None:
        return []
    pattern = PATTERNS.get(pattern_name)
    if pattern and not pattern.match(value):
        return [f"{path}: Invalid format: '{value}'"]
    return []


def validate_evidence(evidence, path):
    errors = []
    if not isinstance(evidence, dict):
        return [f"{path}: Must be an object"]
    if "pmcid" in evidence:
        errors.extend(validate_pattern(evidence["pmcid"], "pmcid", f"{path}.pmcid"))
    if "text" in evidence and len(evidence["text"]) > 500:
        errors.append(f"{path}.text: Exceeds 500 characters")
    return errors


def validate_go_annotation(ann, idx):
    path = f"task1_function[{idx}]"
    errors = []
    has_go_id = "go_id" in ann
    has_description = "description" in ann

    if has_go_id and has_description:
        errors.append(f"{path}: Cannot have both 'go_id' and 'description'")
    elif not has_go_id and not has_description:
        errors.append(f"{path}: Missing 'go_id' or 'description'")
    elif has_go_id:
        errors.extend(validate_pattern(ann["go_id"], "go_id", f"{path}.go_id"))
    else:
        if not isinstance(ann["description"], str):
            errors.append(f"{path}.description: Must be a string")
        elif len(ann["description"]) > 100:
            errors.append(f"{path}.description: Exceeds 100 characters")

    if "qualifier" not in ann:
        errors.append(f"{path}: Missing 'qualifier'")
    elif ann["qualifier"] not in QUALIFIERS:
        errors.append(f"{path}.qualifier: Invalid value '{ann['qualifier']}'")

    if "aspect" not in ann:
        errors.append(f"{path}: Missing 'aspect'")
    elif ann["aspect"] not in ASPECTS:
        errors.append(f"{path}.aspect: Invalid value '{ann['aspect']}'")

    if "is_negated" in ann and not isinstance(ann["is_negated"], bool):
        errors.append(f"{path}.is_negated: Must be boolean")

    if "evidence" in ann:
        errors.extend(validate_evidence(ann["evidence"], f"{path}.evidence"))

    return errors


def validate_expression_record(rec, idx):
    path = f"task2_expression[{idx}]"
    errors = []

    if "expression_type" not in rec:
        errors.append(f"{path}: Missing 'expression_type'")
    elif rec["expression_type"] not in EXPRESSION_TYPES:
        errors.append(f"{path}.expression_type: Invalid value '{rec['expression_type']}'")

    if "anatomy_id" in rec and rec["anatomy_id"]:
        errors.extend(validate_pattern(rec["anatomy_id"], "fbbt_id", f"{path}.anatomy_id"))
    if "stage_id" in rec and rec["stage_id"]:
        errors.extend(validate_pattern(rec["stage_id"], "fbdv_id", f"{path}.stage_id"))

    if "evidence" in rec:
        errors.extend(validate_evidence(rec["evidence"], f"{path}.evidence"))

    return errors


def validate_output(data):
    errors = []

    if "gene_id" not in data:
        errors.append("Missing 'gene_id'")
    else:
        errors.extend(validate_pattern(data["gene_id"], "gene_id", "gene_id"))

    if "gene_symbol" not in data:
        errors.append("Missing 'gene_symbol'")

    if "task1_function" not in data:
        errors.append("Missing 'task1_function'")
    elif not isinstance(data["task1_function"], list):
        errors.append("task1_function: Must be an array")
    else:
        for i, ann in enumerate(data["task1_function"]):
            errors.extend(validate_go_annotation(ann, i))

    if "task2_expression" not in data:
        errors.append("Missing 'task2_expression'")
    elif not isinstance(data["task2_expression"], list):
        errors.append("task2_expression: Must be an array")
    else:
        for i, rec in enumerate(data["task2_expression"]):
            errors.extend(validate_expression_record(rec, i))

    if "task3_synonyms" not in data:
        errors.append("Missing 'task3_synonyms'")
    elif not isinstance(data["task3_synonyms"], dict):
        errors.append("task3_synonyms: Must be an object")
    else:
        syns = data["task3_synonyms"]
        for field_name in ["fullname_synonyms", "symbol_synonyms"]:
            if field_name not in syns:
                errors.append(f"task3_synonyms: Missing '{field_name}'")
            elif not isinstance(syns[field_name], list):
                errors.append(f"task3_synonyms.{field_name}: Must be an array")
            elif not all(isinstance(s, str) for s in syns[field_name]):
                errors.append(f"task3_synonyms.{field_name}: All items must be strings")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate agent output JSON")
    parser.add_argument("file", help="Path to output JSON file")
    args = parser.parse_args()

    try:
        with open(args.file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    errors = validate_output(data)

    if errors:
        print(f"Validation FAILED ({len(errors)} errors):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        # Also print annotation counts
        counts = {}
        if isinstance(data.get("task1_function"), list):
            counts["task1_go_terms"] = len(data["task1_function"])
        if isinstance(data.get("task2_expression"), list):
            counts["task2_expression"] = len(data["task2_expression"])
        if isinstance(data.get("task3_synonyms"), dict):
            syns = data["task3_synonyms"]
            counts["task3_synonyms"] = len(syns.get("fullname_synonyms", [])) + len(syns.get("symbol_synonyms", []))
        print(f"  Annotation counts: {json.dumps(counts)}", file=sys.stderr)
        sys.exit(1)
    else:
        counts = {
            "task1_go_terms": len(data.get("task1_function", [])),
            "task2_expression": len(data.get("task2_expression", [])),
        }
        if isinstance(data.get("task3_synonyms"), dict):
            syns = data["task3_synonyms"]
            counts["task3_synonyms"] = len(syns.get("fullname_synonyms", [])) + len(syns.get("symbol_synonyms", []))
        print(f"Valid. Annotations: {json.dumps(counts)}")


if __name__ == "__main__":
    main()
