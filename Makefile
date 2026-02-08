# Makefile for Drosophila Gene Annotation Benchmark
#
# Usage:
#   make help              - Show available targets
#   make scaling-all       - Run scaling experiments for all 3 methods
#   make scaling-single    - Run Single-Agent scaling (1,2,4,8,16 papers)
#   make scaling-multi     - Run Multi-Agent scaling (1,2,4,8,16 papers)
#   make scaling-pipeline  - Run Pipeline scaling (1,2,4,8,16 papers)

.PHONY: help install lint typecheck test clean \
        scaling-all scaling-single scaling-multi scaling-pipeline scaling-memorization \
        eval eval-nl analyze compare-methods

# Default paper budgets for scaling experiments
PAPERS := 1 2 4 8 16

# Model to use (override with: make scaling-single MODEL=gpt-5)
MODEL := gpt-5-mini

# Number of parallel workers (override with: make scaling-single WORKERS=1 for sequential)
WORKERS := 4

# Verbosity flag
VERBOSE :=

# Output base directory
OUTPUT_BASE := outputs/scaling

# Git commit short hash for output directory naming
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------
help:
	@echo "Drosophila Gene Annotation Benchmark"
	@echo ""
	@echo "Development:"
	@echo "  make install      - Install dependencies with uv"
	@echo "  make lint         - Run ruff linter"
	@echo "  make typecheck    - Run pyright type checker"
	@echo "  make format       - Format code with ruff"
	@echo ""
	@echo "Scaling Experiments (papers: $(PAPERS)):"
	@echo "  make scaling-all         - Run all 3 methods"
	@echo "  make scaling-single      - Single-Agent (sequential tool use)"
	@echo "  make scaling-multi       - Multi-Agent (hierarchical delegation)"
	@echo "  make scaling-pipeline    - Pipeline (fixed parallel DAG)"
	@echo "  make scaling-memorization - Memorization baseline (no literature)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval DIR=outputs/exp/mcp     - Evaluate outputs in DIR"
	@echo "  make eval-nl DIR=outputs/exp/mcp  - NL calibration/quality evaluation"
	@echo "  make analyze DIR=outputs/exp/mcp  - Analyze FP sources in DIR"
	@echo "  make compare-methods SINGLE=... MULTI=... PIPELINE=... MEMORIZATION=..."
	@echo "                                    - Generate 4-method comparison plots"
	@echo ""
	@echo "Options (override with VAR=value):"
	@echo "  MODEL=$(MODEL)       - LLM model"
	@echo "  WORKERS=$(WORKERS)               - Parallel workers (default: 4)"
	@echo "  PAPERS=$(PAPERS)      - Paper budgets"
	@echo "  VERBOSE=-v                - Enable verbose output"

#------------------------------------------------------------------------------
# Development
#------------------------------------------------------------------------------
install:
	uv sync

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

format:
	uv run ruff format

typecheck:
	uv run pyright

check: lint typecheck
	@echo "All checks passed!"

#------------------------------------------------------------------------------
# Scaling Experiments
#------------------------------------------------------------------------------

# Run all three methods
scaling-all: scaling-single scaling-multi scaling-pipeline
	@echo "All scaling experiments complete!"
	@echo "Results in: $(OUTPUT_BASE)/"

# Single-Agent scaling experiment
scaling-single:
	@echo "Running Single-Agent scaling experiment..."
	uv run python -m scripts.run_scaling_experiment \
		--method single \
		--papers $(PAPERS) \
		--all-genes \
		--model $(MODEL) \
		--workers $(WORKERS) \
		$(VERBOSE)

# Multi-Agent scaling experiment
scaling-multi:
	@echo "Running Multi-Agent scaling experiment..."
	uv run python -m scripts.run_scaling_experiment \
		--method multi \
		--papers $(PAPERS) \
		--all-genes \
		--model $(MODEL) \
		--workers $(WORKERS) \
		$(VERBOSE)

# Pipeline scaling experiment
scaling-pipeline:
	@echo "Running Pipeline scaling experiment..."
	uv run python -m scripts.run_scaling_experiment \
		--method pipeline \
		--papers $(PAPERS) \
		--all-genes \
		--model $(MODEL) \
		$(VERBOSE)

# Memorization baseline (no literature access)
scaling-memorization:
	@echo "Running Memorization baseline experiment..."
	uv run python -m scripts.run_scaling_experiment \
		--method memorization \
		--all-genes \
		--model $(MODEL) \
		$(VERBOSE)

# Dry run to see what would execute
scaling-dry-run:
	@echo "=== Single-Agent ==="
	uv run python -m scripts.run_scaling_experiment \
		--method single --papers $(PAPERS) --all-genes --workers $(WORKERS) --dry-run
	@echo ""
	@echo "=== Multi-Agent ==="
	uv run python -m scripts.run_scaling_experiment \
		--method multi --papers $(PAPERS) --all-genes --workers $(WORKERS) --dry-run
	@echo ""
	@echo "=== Pipeline ==="
	uv run python -m scripts.run_scaling_experiment \
		--method pipeline --papers $(PAPERS) --all-genes --dry-run

#------------------------------------------------------------------------------
# Quick experiments (subset of genes for testing)
#------------------------------------------------------------------------------

# Quick scaling with 10 genes (for testing)
scaling-quick:
	@echo "Running quick scaling (10 genes)..."
	uv run python -m scripts.run_scaling_experiment \
		--papers 1 4 16 \
		--max-genes 10 \
		--model $(MODEL) \
		--output-dir $(OUTPUT_BASE)/quick-$(COMMIT) \
		$(VERBOSE)

# Single gene test
test-single:
	uv run python -m agent.run_agent \
		--gene-id FBgn0000014 \
		--gene-symbol abd-A \
		--max-papers 3 \
		-v

test-multi:
	uv run python -m agent.run_agent \
		--gene-id FBgn0000014 \
		--gene-symbol abd-A \
		--max-papers 3 \
		--multi-agent \
		-v

test-pipeline:
	uv run python -m agent.run_pipeline \
		--gene-id FBgn0000014 \
		--gene-symbol abd-A \
		--max-papers 3 \
		-v

# Test memorization baseline (no literature)
test-memorization:
	uv run python -m agent.run_agent \
		--gene-id FBgn0000014 \
		--gene-symbol abd-A \
		--no-literature \
		-v

#------------------------------------------------------------------------------
# Evaluation & Analysis
#------------------------------------------------------------------------------

# Evaluate outputs (requires DIR variable)
eval-nl:
ifndef DIR
	$(error DIR is required. Usage: make eval-nl DIR=outputs/exp/mcp)
endif
	uv run python -m eval.run_nl_eval --output-dir $(DIR) $(VERBOSE)

# Standard evaluation (requires DIR variable)
eval:
ifndef DIR
	$(error DIR is required. Usage: make eval DIR=outputs/exp/mcp)
endif
	uv run python -m eval.run_eval --batch --output-dir $(DIR) -v

# Analyze FP sources (requires DIR variable)
analyze:
ifndef DIR
	$(error DIR is required. Usage: make analyze DIR=outputs/exp/mcp_subagent)
endif
	uv run python -m scripts.analyze_fp_sources --input-dir $(DIR) -v

# Run scaling analysis with plots (single method)
analyze-scaling:
ifndef DIR
	$(error DIR is required. Usage: make analyze-scaling DIR=outputs/scaling/single-abc123)
endif
	uv run python -m scripts.analyze_scaling --base-dir $(DIR) --plot -v

# Generate comparison plots for all methods
# Usage: make compare-methods SINGLE=outputs/scaling/single-xxx MULTI=outputs/scaling/multi-xxx ...
compare-methods:
ifndef SINGLE
	$(error SINGLE is required. Usage: make compare-methods SINGLE=... MULTI=... PIPELINE=... MEMORIZATION=...)
endif
	uv run python -m scripts.analyze_scaling \
		--compare \
		--single-dir $(SINGLE) \
		$(if $(MULTI),--multi-dir $(MULTI)) \
		$(if $(PIPELINE),--pipeline-dir $(PIPELINE)) \
		$(if $(MEMORIZATION),--memorization-dir $(MEMORIZATION)) \
		--papers 0 1 2 4 8 16 \
		--metric recall --combined --plot
	uv run python -m scripts.analyze_scaling \
		--compare \
		--single-dir $(SINGLE) \
		$(if $(MULTI),--multi-dir $(MULTI)) \
		$(if $(PIPELINE),--pipeline-dir $(PIPELINE)) \
		$(if $(MEMORIZATION),--memorization-dir $(MEMORIZATION)) \
		--papers 0 1 2 4 8 16 \
		--metric recall --in-corpus --combined --plot
	uv run python -m scripts.analyze_scaling \
		--compare \
		--single-dir $(SINGLE) \
		$(if $(MULTI),--multi-dir $(MULTI)) \
		$(if $(PIPELINE),--pipeline-dir $(PIPELINE)) \
		$(if $(MEMORIZATION),--memorization-dir $(MEMORIZATION)) \
		--papers 0 1 2 4 8 16 \
		--metric weighted_recall --combined --plot
	uv run python -m scripts.analyze_scaling \
		--compare \
		--single-dir $(SINGLE) \
		$(if $(MULTI),--multi-dir $(MULTI)) \
		$(if $(PIPELINE),--pipeline-dir $(PIPELINE)) \
		$(if $(MEMORIZATION),--memorization-dir $(MEMORIZATION)) \
		--papers 0 1 2 4 8 16 \
		--metric weighted_recall --in-corpus --combined --plot
	@echo "Comparison plots saved to: $(OUTPUT_BASE)/comparison_plots/"

#------------------------------------------------------------------------------
# Cleanup
#------------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache 2>/dev/null || true

clean-outputs:
	@echo "This will delete all outputs. Are you sure? [y/N]"
	@read ans && [ "$$ans" = "y" ] && rm -rf outputs/* || echo "Aborted."
