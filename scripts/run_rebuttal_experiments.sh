#!/bin/bash
# Rebuttal experiments for KDD 2026 Submission 262
#
# Prerequisites:
#   1. Install litellm: pip install litellm[proxy]
#   2. Set env vars:
#      - OPENAI_API_KEY (for E3)
#      - AWS_BEARER_TOKEN_BEDROCK (for E1, E2 — from bedrock API key CSV)
#   3. Start litellm proxy in a separate terminal:
#      litellm --config litellm_config.yaml --port 4000
#
# Usage:
#   ./scripts/run_rebuttal_experiments.sh [e1|e2|e3|all]

set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENT="${1:-all}"
WORKERS=4

run_e1() {
    echo "=== E1: Claude Sonnet 4 (Bedrock) ==="
    export OPENAI_BASE_URL=http://localhost:4000

    uv run python -m scripts.run_experiment_auto \
        --method memorization --model claude-sonnet-4 \
        --all-genes --name "memorization-claude-sonnet4"

    uv run python -m scripts.run_experiment_auto \
        --method single --papers 16 --model claude-sonnet-4 \
        --workers "$WORKERS" --all-genes --name "single-claude-sonnet4-16p"

    uv run python -m scripts.run_experiment_auto \
        --method multi --papers 16 --model claude-sonnet-4 \
        --workers "$WORKERS" --all-genes --name "multi-claude-sonnet4-16p"
}

run_e2() {
    echo "=== E2: DeepSeek V3.2 (Bedrock) ==="
    export OPENAI_BASE_URL=http://localhost:4000

    uv run python -m scripts.run_experiment_auto \
        --method memorization --model deepseek-v3.2 \
        --all-genes --name "memorization-deepseek-v3.2"

    uv run python -m scripts.run_experiment_auto \
        --method single --papers 16 --model deepseek-v3.2 \
        --workers "$WORKERS" --all-genes --name "single-deepseek-v3.2-16p"

    uv run python -m scripts.run_experiment_auto \
        --method multi --papers 16 --model deepseek-v3.2 \
        --workers "$WORKERS" --all-genes --name "multi-deepseek-v3.2-16p"
}

run_e3() {
    echo "=== E3: GPT-5-mini Multi-Agent budget 32 ==="
    unset OPENAI_BASE_URL 2>/dev/null || true

    uv run python -m scripts.run_experiment_auto \
        --method multi --papers 32 --model gpt-5-mini \
        --workers "$WORKERS" --all-genes --name "multi-gpt5mini-32p"
}

case "$EXPERIMENT" in
    e1) run_e1 ;;
    e2) run_e2 ;;
    e3) run_e3 ;;
    all)
        run_e1
        run_e2
        run_e3
        ;;
    *)
        echo "Usage: $0 [e1|e2|e3|all]"
        exit 1
        ;;
esac

echo "=== Done ==="
