#!/usr/bin/env python3
"""Autonomous experiment runner with retry logic and auto-reporting.

Wraps the existing scaling experiment infrastructure with:
- Pre-flight validation (imports, API keys, ontology indices)
- Automatic retry on transient failures (rate limits, timeouts)
- Post-run evaluation and Markdown report generation
- Resume support (inherited from run_scaling_experiment)

Usage:
    # Full autonomous run: experiment + eval + report
    uv run python -m scripts.run_experiment_auto \
        --method single --papers 8 16 --workers 4

    # Dry run (validate environment, show plan)
    uv run python -m scripts.run_experiment_auto --dry-run

    # Skip eval/report (just run the experiment)
    uv run python -m scripts.run_experiment_auto --no-report

    # Custom name for the report
    uv run python -m scripts.run_experiment_auto --name "single-agent-v2"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def validate_environment() -> list[str]:
    """Check that the environment is ready to run experiments.

    Returns:
        Tuple of (errors, warnings). Errors are fatal; warnings are informational.
    """
    errors = []
    warnings = []

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        # Try loading from .env
        try:
            from dotenv import load_dotenv

            load_dotenv(override=True)
        except ImportError:
            pass
        if not os.environ.get("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY not set. Check .env file.")

    # Check imports
    try:
        from agent.agentic import run_agent_mcp  # noqa: F401
    except ImportError as e:
        errors.append(f"Agent import failed: {e}. Run 'uv sync'.")

    try:
        from eval.run_eval import main  # noqa: F401
    except ImportError as e:
        warnings.append(f"Eval import failed: {e}. Evaluation step will be skipped.")

    # Check data files
    data_dir = Path("data")
    required_files = ["genes_top100.csv", "ground_truth_top100.jsonl", "gene_to_pmcids_top100.json"]
    for f in required_files:
        if not (data_dir / f).exists():
            errors.append(f"Missing data file: data/{f}")

    # Check ontology index
    ontology_dir = Path("ontologies")
    if not ontology_dir.exists() or not any(ontology_dir.iterdir()):
        errors.append("Ontology indices missing. Run: uv run python -m scripts.download_ontologies")

    return errors, warnings


def run_experiment_with_retry(
    method: str,
    papers: list[int],
    max_genes: int | None,
    all_genes: bool,
    model: str,
    workers: int,
    verbose: bool,
    output_dir: Path | None,
    hide_terms: bool,
    oracle_retrieval: bool = False,
    max_retries: int = 2,
) -> Path:
    """Run the scaling experiment with retry on failure.

    Returns:
        Path to the experiment output directory.
    """
    cmd = [
        "uv", "run", "python", "-m", "scripts.run_scaling_experiment",
        "--method", method,
        "--papers", *[str(p) for p in papers],
        "--model", model,
        "--workers", str(workers),
    ]

    if all_genes:
        cmd.append("--all-genes")
    if max_genes:
        cmd.extend(["--max-genes", str(max_genes)])
    if verbose:
        cmd.append("-v")
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    if hide_terms:
        cmd.append("--hide-terms")
    if oracle_retrieval:
        cmd.append("--oracle-retrieval")

    # Determine output directory (match run_scaling_experiment logic)
    if output_dir is None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit = "unknown"
        output_dir = Path(f"outputs/scaling/{method}-{commit}")

    for attempt in range(max_retries + 1):
        print(f"\n{'=' * 60}")
        if attempt > 0:
            print(f"RETRY {attempt}/{max_retries}")
            # Back off before retry
            wait = min(30 * (2 ** (attempt - 1)), 120)
            print(f"Waiting {wait}s before retry...")
            time.sleep(wait)

        print(f"Running: {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd, cwd=Path.cwd())

        if result.returncode == 0:
            print(f"\nExperiment completed successfully. Output: {output_dir}")
            return output_dir

        # Check if we got partial results (resume will handle it)
        summary_path = output_dir / "experiment_summary.json"
        if summary_path.exists():
            print(f"\nExperiment had errors but produced partial results at {output_dir}")
            print("Re-running will resume from where it left off.")
            if attempt < max_retries:
                continue
            else:
                print("Max retries reached. Proceeding with partial results.")
                return output_dir

        if attempt < max_retries:
            print(f"\nExperiment failed (exit code {result.returncode}). Will retry.")
        else:
            print(f"\nExperiment failed after {max_retries + 1} attempts.")
            sys.exit(1)

    return output_dir  # unreachable but satisfies type checker


def run_evaluation(output_dir: Path, papers: list[int], verbose: bool) -> dict:
    """Run evaluation on experiment results.

    Returns:
        Dict mapping paper_config -> eval results path.
    """
    eval_results = {}

    for p in papers:
        paper_dir = output_dir / f"papers_{p}"
        if not paper_dir.exists():
            print(f"Skipping eval for papers_{p} (directory not found)")
            continue

        print(f"\nEvaluating papers_{p}...")
        cmd = ["uv", "run", "python", "-m", "eval.run_eval", "--batch", "--output-dir", str(paper_dir)]
        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)

        if result.returncode == 0:
            eval_results[p] = str(paper_dir / "eval_results.json")
            print(f"  Eval complete for papers_{p}")
        else:
            print(f"  Eval failed for papers_{p}: {result.stderr[-500:]}")

    return eval_results


def generate_report(
    output_dir: Path,
    method: str,
    model: str,
    papers: list[int],
    eval_results: dict,
    name: str | None = None,
) -> Path:
    """Generate a Markdown experiment report.

    Returns:
        Path to the report file.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_name = name or f"{method}-{model}-{timestamp}"
    report_path = reports_dir / f"{report_name}.md"

    # Load experiment summary
    summary_path = output_dir / "experiment_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    # Build report
    lines = [
        f"# Experiment Report: {report_name}",
        "",
        f"**Date:** {timestamp}",
        f"**Method:** {method}",
        f"**Model:** {model}",
        f"**Paper configs:** {papers}",
        f"**Output directory:** `{output_dir}`",
        "",
    ]

    # Summary table from experiment_summary.json
    if summary.get("results_by_config"):
        lines.extend([
            "## Experiment Summary",
            "",
            "| Papers | Successful | Failed | Total Cost | Avg Papers Read |",
            "|--------|-----------|--------|------------|-----------------|",
        ])
        for config_key, config_data in sorted(summary["results_by_config"].items(), key=lambda x: int(x[0])):
            lines.append(
                f"| {config_key} | {config_data['successful']} | {config_data['failed']} "
                f"| ${config_data['total_cost']:.4f} | {config_data.get('avg_papers_read', 'N/A'):.1f} |"
            )
        lines.extend(["", f"**Total cost:** ${summary.get('total_cost', 0):.4f}", ""])

    # Eval results
    if eval_results:
        lines.extend(["## Evaluation Results", ""])
        for p, eval_path in sorted(eval_results.items()):
            if Path(eval_path).exists():
                with open(eval_path) as f:
                    eval_data = json.load(f)
                lines.append(f"### Papers = {p}")
                lines.append("")

                # Extract aggregate metrics if available
                if isinstance(eval_data, dict):
                    agg = eval_data.get("aggregate", eval_data)
                    for task_name, task_data in agg.items():
                        if isinstance(task_data, dict):
                            lines.append(f"- **{task_name}**: {json.dumps(task_data, indent=None)}")
                lines.append("")

    # Failure analysis
    if summary.get("results_by_config"):
        failed_genes = []
        for _config_key, config_data in summary["results_by_config"].items():
            if config_data.get("failed", 0) > 0:
                failed_genes.append(f"papers={_config_key}: {config_data['failed']} failures")
        if failed_genes:
            lines.extend([
                "## Failure Analysis",
                "",
                *[f"- {fg}" for fg in failed_genes],
                "",
                "*Note: These are observations. Root cause analysis of individual failures "
                "requires examining the trace files in the output directory.*",
                "",
            ])

    # Gene list
    if summary.get("genes"):
        lines.extend([
            "## Genes Tested",
            "",
            f"Total: {len(summary['genes'])} genes",
            "",
        ])

    lines.extend([
        "## Key Findings",
        "",
        "*TODO: Fill in after reviewing results.*",
        "",
        "---",
        f"*Report generated automatically by `run_experiment_auto.py` on {timestamp}.*",
    ])

    report_path.write_text("\n".join(lines))
    print(f"\nReport written to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous experiment runner: run + evaluate + report"
    )
    parser.add_argument("--method", "-m", default="single", choices=["single", "multi", "pipeline", "memorization"])
    parser.add_argument("--papers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--max-genes", type=int)
    parser.add_argument("--all-genes", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output-dir", "-o", type=Path)
    parser.add_argument("--hide-terms", action="store_true")
    parser.add_argument("--oracle-retrieval", action="store_true", help="Oracle retrieval (ground truth papers)")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries on failure (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Validate environment and show plan")
    parser.add_argument("--no-report", action="store_true", help="Skip evaluation and report generation")
    parser.add_argument("--name", type=str, help="Custom name for the report")
    args = parser.parse_args()

    print("=" * 60)
    print("FlyBench Autonomous Experiment Runner")
    print("=" * 60)

    # Step 1: Validate
    print("\n[1/4] Validating environment...")
    errors, warnings = validate_environment()
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    if errors:
        print("Environment validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("  All checks passed.")

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute:")
        print(f"  Method: {args.method}")
        print(f"  Model: {args.model}")
        print(f"  Papers: {args.papers}")
        print(f"  Workers: {args.workers}")
        print(f"  Max genes: {args.max_genes or 'all 25 test genes'}")
        print(f"  All genes: {args.all_genes}")
        print(f"  Max retries: {args.max_retries}")
        return

    # Step 2: Run experiment
    print(f"\n[2/4] Running {args.method} experiment...")
    output_dir = run_experiment_with_retry(
        method=args.method,
        papers=args.papers,
        max_genes=args.max_genes,
        all_genes=args.all_genes,
        model=args.model,
        workers=args.workers,
        verbose=args.verbose,
        output_dir=args.output_dir,
        hide_terms=args.hide_terms,
        oracle_retrieval=args.oracle_retrieval,
        max_retries=args.max_retries,
    )

    if args.no_report:
        print("\nDone (skipping eval and report).")
        return

    # Step 3: Evaluate
    print(f"\n[3/4] Evaluating results...")
    eval_results = run_evaluation(output_dir, args.papers, args.verbose)

    # Step 4: Report
    print(f"\n[4/4] Generating report...")
    report_path = generate_report(
        output_dir=output_dir,
        method=args.method,
        model=args.model,
        papers=args.papers,
        eval_results=eval_results,
        name=args.name,
    )

    print(f"\n{'=' * 60}")
    print("AUTONOMOUS RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output:  {output_dir}")
    print(f"  Report:  {report_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the report: cat {report_path}")
    print(f"  2. Fill in 'Key Findings' section")
    print(f"  3. Commit: git add {report_path} && git commit -m 'docs: add {report_path.stem} report'")


if __name__ == "__main__":
    main()
