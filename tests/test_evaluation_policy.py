import numpy as np

from flyaoc.data import load_benchmark_records
from flyaoc.evaluation.evaluate import aggregate_gene_results
from flyaoc.reporting.tables import _bootstrap_intervals, _micro_values


def test_verified_fact_denominators_match_current_benchmark() -> None:
    records = load_benchmark_records()

    task1 = 0
    task2 = 0
    task3 = 0
    for record in records:
        task1 += len(
            {
                item["go_id"]
                for item in record["task1_function"]
                if item.get("go_id")
                and item.get("in_corpus_verified", item.get("in_corpus", False))
            }
        )
        task2 += len(
            {
                item["anatomy_id"]
                for item in record["task2_expression"]
                if item.get("anatomy_id")
                and item.get("in_corpus_verified", item.get("in_corpus", False))
            }
        )
        task3 += len(
            {
                item["synonym"].strip().lower()
                for group in ("fullname_synonyms", "symbol_synonyms")
                for item in record["task3_synonyms"][group]
                if item.get("synonym")
                and item.get("in_corpus_verified", item.get("in_corpus", False))
            }
        )

    assert task1 == 770
    assert task2 == 252
    assert task3 == 457


def test_failed_or_empty_runs_contribute_zero_recall_to_micro_denominator() -> None:
    ok_gene = {
        "task1_function": {
            "gt_verified_count": 2,
            "primary_k": 30,
            "semantic_recall_at_k": {"20": 0.5, "30": 0.5},
            "semantic_recall_at_30": 0.5,
            "semantic_recall_sum_at_30": 1.0,
        },
        "task2_expression": {
            "gt_verified_count": 2,
            "anatomy_semantic_recall_at_k": {"10": 1.0},
            "anatomy_semantic_recall_at_10": 1.0,
            "anatomy_semantic_recall_sum_at_10": 2.0,
        },
        "task3_synonyms": {
            "gt_verified_combined_count": 2,
            "combined_exact_recall_at_k": {"20": 0.5},
            "combined_exact_recall_at_20": 0.5,
            "combined_exact_hits_at_20": 1,
        },
    }
    failed_gene = {
        "task1_function": {
            "gt_verified_count": 2,
            "primary_k": 30,
            "semantic_recall_at_k": {"20": 0.0, "30": 0.0},
            "semantic_recall_at_30": 0.0,
            "semantic_recall_sum_at_30": 0.0,
        },
        "task2_expression": {
            "gt_verified_count": 2,
            "anatomy_semantic_recall_at_k": {"10": 0.0},
            "anatomy_semantic_recall_at_10": 0.0,
            "anatomy_semantic_recall_sum_at_10": 0.0,
        },
        "task3_synonyms": {
            "gt_verified_combined_count": 2,
            "combined_exact_recall_at_k": {"20": 0.0},
            "combined_exact_recall_at_20": 0.0,
            "combined_exact_hits_at_20": 0,
        },
    }

    aggregate = aggregate_gene_results([ok_gene, failed_gene])

    assert aggregate["task1_verified_fact_count"] == 4
    assert aggregate["task2_verified_fact_count"] == 4
    assert aggregate["task3_verified_fact_count"] == 4
    assert aggregate["task1_semantic_recall_at_30_micro"] == 0.25
    assert aggregate["task2_anatomy_semantic_recall_at_10_micro"] == 0.5
    assert aggregate["task3_combined_exact_recall_at_20_micro"] == 0.25


def test_bootstrap_resamples_genes_and_recomputes_micro_estimator() -> None:
    contributions = [
        {
            "go_num": 1.0,
            "go_den": 2,
            "expr_num": 2.0,
            "expr_den": 4,
            "syn_num": 1,
            "syn_den": 2,
        },
        {
            "go_num": 3.0,
            "go_den": 6,
            "expr_num": 1.0,
            "expr_den": 2,
            "syn_num": 3,
            "syn_den": 6,
        },
    ]

    values = _micro_values(contributions)
    intervals = _bootstrap_intervals(
        contributions,
        n_bootstrap=50,
        rng=np.random.default_rng(123),
    )

    assert values["go"] == 0.5
    assert values["expr"] == 0.5
    assert values["syn"] == 0.5
    assert values["avg"] == 0.5
    assert intervals["avg"] == (0.5, 0.5)
