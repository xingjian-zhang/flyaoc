#!/usr/bin/env python3
"""Host-side orchestrator: run Claude Code on each gene in a fresh Docker container.

Usage:
    # Run all 10 genes
    python3 baselines/claude_code/run_benchmark.py

    # Run specific genes
    python3 baselines/claude_code/run_benchmark.py --gene-ids FBgn0284256 FBgn0002736

    # Resume (skip genes that already have output)
    python3 baselines/claude_code/run_benchmark.py --resume

    # Custom timeout
    python3 baselines/claude_code/run_benchmark.py --timeout 2400

Prerequisites:
    1. Docker image built: docker build -t flybench-cc baselines/claude_code/
    2. Papers split: python3 baselines/claude_code/split_papers.py
    3. Claude Code authenticated (credentials in ~/.claude/.credentials.json)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths relative to drosophila-agent-benchmark/
BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_DIR = BENCHMARK_ROOT / "baselines" / "claude_code"
GENES_FILE = BASELINE_DIR / "genes.json"
PAPERS_SPLIT_DIR = BASELINE_DIR / "papers_split"

# Data paths to volume-mount
CORPUS_CACHE_DIR = BENCHMARK_ROOT / "corpus_cache"
ONTOLOGIES_DIR = BENCHMARK_ROOT / "ontologies"
ONTOLOGY_INDICES_DIR = BENCHMARK_ROOT / "ontology_indices"
GENE_DATA_DIR = BENCHMARK_ROOT / "data"

# Output directory
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / "outputs" / "claude_code"

# Docker image name
DOCKER_IMAGE = "flybench-cc:latest"

# Default timeout per gene (seconds)
DEFAULT_TIMEOUT = 1800  # 30 minutes

# Default paper budget (matches Multi-Agent BudgetConfig default)
DEFAULT_MAX_PAPERS = 10


def load_genes(gene_ids: list[str] | None = None) -> list[dict]:
    """Load genes from genes.json, optionally filtered by IDs."""
    with open(GENES_FILE) as f:
        genes = json.load(f)
    if gene_ids:
        genes = [g for g in genes if g["gene_id"] in gene_ids]
        found = {g["gene_id"] for g in genes}
        missing = set(gene_ids) - found
        if missing:
            print(f"Warning: genes not found in genes.json: {missing}", file=sys.stderr)
    return genes


def check_prerequisites():
    """Verify that all required files and directories exist."""
    errors = []

    if not PAPERS_SPLIT_DIR.exists():
        errors.append(
            f"Papers not split. Run: python3 baselines/claude_code/split_papers.py"
        )

    if not CORPUS_CACHE_DIR.exists():
        errors.append(f"Corpus cache not found: {CORPUS_CACHE_DIR}")

    if not (CORPUS_CACHE_DIR / "bm25_index.pkl").exists():
        errors.append(f"BM25 index not found: {CORPUS_CACHE_DIR / 'bm25_index.pkl'}")

    if not ONTOLOGIES_DIR.exists():
        errors.append(f"Ontologies directory not found: {ONTOLOGIES_DIR}")

    if not ONTOLOGY_INDICES_DIR.exists():
        errors.append(f"Ontology indices not found: {ONTOLOGY_INDICES_DIR}")

    # Check Docker image exists
    result = subprocess.run(
        ["docker", "image", "inspect", DOCKER_IMAGE],
        capture_output=True,
    )
    if result.returncode != 0:
        errors.append(
            f"Docker image not found. Build with: docker build -t flybench-cc baselines/claude_code/"
        )

    # Check Claude credentials
    local_creds = BASELINE_DIR / ".credentials.json"
    home_creds = Path.home() / ".claude" / ".credentials.json"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not local_creds.exists() and not home_creds.exists() and not api_key:
        errors.append(
            f"Claude credentials not found. Either:\n"
            f"    - Copy ~/.claude/.credentials.json to {local_creds}\n"
            f"    - Or set ANTHROPIC_API_KEY environment variable"
        )

    if errors:
        print("Prerequisites check failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def build_prompt(gene: dict) -> str:
    """Build the prompt for Claude Code."""
    return (
        f"Annotate the Drosophila gene **{gene['gene_symbol']}** ({gene['gene_id']}).\n\n"
        f"**Gene Summary:** {gene['summary']}\n\n"
        f"Follow the instructions in /app/CLAUDE.md. "
        f"Write your output to /output/{gene['gene_id']}.json. "
        f"After writing, validate with: python3 /app/scripts/validate_output.py /output/{gene['gene_id']}.json\n\n"
        f"Be thorough — read multiple papers, extract ALL annotations across all 3 tasks, "
        f"and use ontology search to find correct term IDs."
    )


def run_gene(
    gene: dict,
    output_dir: Path,
    log_dir: Path,
    timeout: int,
    model: str,
    max_turns: int,
    max_papers: int,
) -> dict:
    """Run Claude Code on a single gene in a Docker container.

    Returns a dict with gene_id, success, duration, output_file, error.
    """
    gene_id = gene["gene_id"]
    gene_symbol = gene["gene_symbol"]
    prompt = build_prompt(gene)

    # Ensure output and log dirs exist
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{gene_id}.json"
    log_file = log_dir / f"{gene_id}.log"
    transcript_file = log_dir / f"{gene_id}_transcript.jsonl"

    print(f"\n{'='*60}")
    print(f"Gene: {gene_symbol} ({gene_id}) | difficulty: {gene.get('difficulty', '?')}")
    print(f"Output: {output_file}")
    print(f"Log: {log_file}")
    print(f"{'='*60}")

    # Find credentials: prefer local copy, fall back to home dir
    local_creds = BASELINE_DIR / ".credentials.json"
    home_creds = Path.home() / ".claude" / ".credentials.json"
    creds_path = local_creds if local_creds.exists() else home_creds

    # Build docker command
    docker_cmd = [
        "docker", "run", "--rm",
        # Use host network so container can reach API endpoints
        "--network=host",
        # Volume mounts (read-only except output)
        "-v", f"{CORPUS_CACHE_DIR}:/data/corpus_cache:ro",
        "-v", f"{PAPERS_SPLIT_DIR}:/data/papers:ro",
        "-v", f"{ONTOLOGIES_DIR}:/data/ontologies:ro",
        "-v", f"{ONTOLOGY_INDICES_DIR}:/data/ontology_indices:ro",
        "-v", f"{output_dir}:/output",
    ]

    # Mount credentials if available
    if creds_path.exists():
        docker_cmd.extend([
            "-v", f"{creds_path}:/home/agent/.claude/.credentials.json:ro",
        ])

    # Environment
    docker_cmd.extend([
        "-e", "DISABLE_AUTOUPDATER=1",
        "-e", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
    ])

    # Paper budget (0 = unlimited)
    if max_papers > 0:
        docker_cmd.extend(["-e", f"MAX_PAPERS={max_papers}"])

    # Pass ANTHROPIC_API_KEY if set (alternative to credentials file)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        docker_cmd.extend(["-e", f"ANTHROPIC_API_KEY={api_key}"])

    # Forward proxy settings for sandboxed environments
    for env_var in ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                    "http_proxy", "https_proxy", "no_proxy"]:
        val = os.environ.get(env_var)
        if val:
            docker_cmd.extend(["-e", f"{env_var}={val}"])

    # Mount proxy CA cert if present (copy to project dir to avoid Docker path restrictions)
    ca_cert_src = os.environ.get("SSL_CERT_FILE")
    ca_cert_local = BASELINE_DIR / "proxy-ca.crt"
    if ca_cert_src and Path(ca_cert_src).exists():
        # Copy cert to a Docker-mountable location if needed
        if not ca_cert_local.exists():
            import shutil
            shutil.copy2(ca_cert_src, ca_cert_local)
        ca_cert_container = "/app/proxy-ca.crt"
        docker_cmd.extend([
            "-v", f"{ca_cert_local}:{ca_cert_container}:ro",
            "-e", f"SSL_CERT_FILE={ca_cert_container}",
            "-e", f"NODE_EXTRA_CA_CERTS={ca_cert_container}",
            "-e", f"REQUESTS_CA_BUNDLE={ca_cert_container}",
        ])

    docker_cmd.extend([
        # Working directory
        "-w", "/app",
        # Image
        DOCKER_IMAGE,
        # Claude Code args
        "--dangerously-skip-permissions",
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
    ])

    # max_turns: -1 means unlimited (rely on timeout), otherwise pass the flag
    if max_turns > 0:
        docker_cmd.extend(["--max-turns", str(max_turns)])

    docker_cmd.extend(["-p", prompt])

    start_time = time.time()

    try:
        with open(log_file, "w") as lf, open(transcript_file, "w") as tf:
            lf.write(f"# Gene: {gene_symbol} ({gene_id})\n")
            lf.write(f"# Started: {datetime.now().isoformat()}\n")
            lf.write(f"# Timeout: {timeout}s\n")
            lf.write(f"# Model: {model}\n")
            lf.write(f"# Command: {' '.join(docker_cmd)}\n\n")

            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output to both console and log file
            for line in process.stdout:
                lf.write(line)
                lf.flush()
                # Write JSON lines to transcript
                stripped = line.strip()
                if stripped.startswith("{"):
                    tf.write(stripped + "\n")
                    tf.flush()
                    # Parse for display
                    try:
                        msg = json.loads(stripped)
                        if msg.get("type") == "assistant":
                            content = msg.get("message", {}).get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text.strip():
                                        # Print first 200 chars of each text block
                                        preview = text[:200] + ("..." if len(text) > 200 else "")
                                        print(f"  CC: {preview}")
                                elif isinstance(block, dict) and block.get("type") == "tool_use":
                                    tool = block.get("name", "?")
                                    print(f"  CC: [tool: {tool}]")
                        elif msg.get("type") == "result":
                            cost = msg.get("cost_usd", 0)
                            turns = msg.get("num_turns", 0)
                            print(f"  Result: {turns} turns, ${cost:.4f}")
                    except json.JSONDecodeError:
                        pass
                else:
                    # Non-JSON output (e.g., stderr)
                    if stripped:
                        print(f"  [{stripped[:120]}]")

            process.wait(timeout=timeout)
            exit_code = process.returncode

    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start_time
        print(f"  TIMEOUT after {duration:.0f}s")
        return {
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "success": False,
            "duration": duration,
            "error": f"Timeout after {timeout}s",
        }
    except Exception as e:
        duration = time.time() - start_time
        print(f"  ERROR: {e}")
        return {
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
            "success": False,
            "duration": duration,
            "error": str(e),
        }

    duration = time.time() - start_time

    # Check if output file was created
    success = output_file.exists()
    if success:
        # Quick validation
        try:
            with open(output_file) as f:
                data = json.load(f)
            task1_count = len(data.get("task1_function", []))
            task2_count = len(data.get("task2_expression", []))
            task3_count = 0
            syns = data.get("task3_synonyms", {})
            if isinstance(syns, dict):
                task3_count = len(syns.get("fullname_synonyms", [])) + len(
                    syns.get("symbol_synonyms", [])
                )
            print(
                f"  OK: {task1_count} GO terms, {task2_count} expression, "
                f"{task3_count} synonyms ({duration:.0f}s)"
            )
        except Exception as e:
            print(f"  WARNING: Output exists but invalid JSON: {e}")
            success = False
    else:
        print(f"  FAILED: No output file created ({duration:.0f}s, exit={exit_code})")

    return {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "success": success,
        "duration": duration,
        "exit_code": exit_code,
        "output_file": str(output_file) if success else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude Code baseline on FlyBench genes"
    )
    parser.add_argument(
        "--gene-ids", nargs="+",
        help="Specific gene IDs to run (default: all in genes.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout per gene in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip genes that already have output files",
    )
    parser.add_argument(
        "--model", default="sonnet",
        help="Claude model to use (default: sonnet)",
    )
    parser.add_argument(
        "--max-turns", type=int, default=-1,
        help="Max agentic turns per gene (-1 = unlimited, default: -1)",
    )
    parser.add_argument(
        "--max-papers", type=int, default=DEFAULT_MAX_PAPERS,
        help=f"Max papers to read per gene (0 = unlimited, default: {DEFAULT_MAX_PAPERS})",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip prerequisite checks",
    )
    args = parser.parse_args()

    if not args.skip_checks:
        check_prerequisites()

    genes = load_genes(args.gene_ids)
    if not genes:
        print("No genes to process", file=sys.stderr)
        sys.exit(1)

    log_dir = args.output_dir / "logs"

    # Resume support
    if args.resume:
        genes = [
            g for g in genes
            if not (args.output_dir / f"{g['gene_id']}.json").exists()
        ]
        if not genes:
            print("All genes already have output files. Nothing to do.")
            sys.exit(0)

    print(f"FlyBench Claude Code Baseline")
    papers_str = str(args.max_papers) if args.max_papers > 0 else "unlimited"
    print(f"Genes: {len(genes)} | Timeout: {args.timeout}s | Model: {args.model} | Papers: {papers_str}")
    print(f"Output: {args.output_dir}")
    print(f"Genes: {', '.join(g['gene_symbol'] for g in genes)}")

    results = []
    for gene in genes:
        result = run_gene(
            gene,
            output_dir=args.output_dir,
            log_dir=log_dir,
            timeout=args.timeout,
            model=args.model,
            max_turns=args.max_turns,
            max_papers=args.max_papers,
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    succeeded = sum(1 for r in results if r["success"])
    total_duration = sum(r["duration"] for r in results)
    print(f"Succeeded: {succeeded}/{len(results)}")
    print(f"Total time: {total_duration:.0f}s ({total_duration/60:.1f}min)")
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  {status}: {r['gene_symbol']} ({r['gene_id']}) - {r['duration']:.0f}s")
        if not r["success"] and r.get("error"):
            print(f"        Error: {r['error']}")

    # Save results summary
    summary_file = args.output_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "max_turns": args.max_turns,
                "max_papers": args.max_papers,
                "timeout": args.timeout,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved: {summary_file}")

    # Exit with error if any failed
    if succeeded < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
