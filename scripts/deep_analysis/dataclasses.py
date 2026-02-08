"""Data classes for deep analysis of scaling experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneTrajectory:
    """Per-gene F1 trajectory across paper configurations."""

    gene_id: str
    gene_symbol: str
    task1_f1: dict[int, float] = field(default_factory=dict)
    task1_precision: dict[int, float] = field(default_factory=dict)
    task1_recall: dict[int, float] = field(default_factory=dict)
    task1_tp: dict[int, int] = field(default_factory=dict)
    task1_fp: dict[int, int] = field(default_factory=dict)
    task2_f1: dict[int, float] = field(default_factory=dict)
    task2_precision: dict[int, float] = field(default_factory=dict)
    task2_recall: dict[int, float] = field(default_factory=dict)
    task3_f1: dict[int, float] = field(default_factory=dict)
    gt_task1_total: int = 0
    gt_task1_in_corpus: int = 0
    gt_task2_total: int = 0
    gt_task2_in_corpus: int = 0
    gt_task3_symbols: int = 0

    def scaling_gain(self, task: str) -> float:
        """Compute F1 gain from min to max papers."""
        f1_dict = getattr(self, f"{task}_f1")
        if not f1_dict:
            return 0.0
        configs = sorted(f1_dict.keys())
        if len(configs) < 2:
            return 0.0
        return f1_dict[configs[-1]] - f1_dict[configs[0]]

    def best_config(self, task: str) -> int:
        """Find the config with highest F1."""
        f1_dict = getattr(self, f"{task}_f1")
        if not f1_dict:
            return 1
        return max(f1_dict.keys(), key=lambda k: f1_dict[k])

    def category(self, task: str, threshold: float = 0.10) -> str:
        """Categorize gene scaling behavior."""
        gain = self.scaling_gain(task)
        if gain > threshold:
            return "improves"
        elif gain < -threshold:
            return "degrades"
        else:
            return "stable"


@dataclass
class FalsePositive:
    """A false positive annotation with full context."""

    gene_id: str
    gene_symbol: str
    predicted_id: str
    predicted_term_name: str | None
    task: str
    similarity: float
    paper_config: int
    evidence_pmcid: str | None = None
    evidence_text: str | None = None
    aspect: str | None = None  # P, F, C for GO


@dataclass
class TruePositive:
    """A true positive annotation with full context."""

    gene_id: str
    gene_symbol: str
    predicted_id: str
    matched_gt_id: str
    similarity: float
    is_exact: bool
    paper_config: int
    evidence_pmcid: str | None = None
    aspect: str | None = None


@dataclass
class ToolCall:
    """A single tool call from trace."""

    tool: str
    args: dict
    result: str | None
    timestamp: str
    gene_id: str
    paper_config: int


@dataclass
class PaperRead:
    """Information about a paper the agent read."""

    pmcid: str
    gene_id: str
    paper_config: int
    read_order: int  # 1st, 2nd, 3rd paper read
    gene_in_title: bool
    relevance_score: float
    annotations_extracted: int = 0
    tp_count: int = 0
    fp_count: int = 0
