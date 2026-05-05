"""Recompute table-ready metrics from normalized frozen predictions."""

from __future__ import annotations

from pathlib import Path

from flyaoc.reporting.tables import (
    evaluate_cross_family_bootstrap,
    evaluate_main_architecture_bootstrap,
    evaluate_model_comparison_bootstrap,
    evaluate_prediction_artifacts,
    manifest_rows_from_evaluations,
    write_csv,
    write_markdown,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts/predictions/manifest.json"
    evaluations = evaluate_prediction_artifacts(manifest)
    rows = manifest_rows_from_evaluations(evaluations)
    write_csv(rows, root / "artifacts/tables/main_results.csv")
    write_markdown(rows, root / "artifacts/tables/main_results.md")

    main_ci_rows = evaluate_main_architecture_bootstrap(
        manifest,
        evaluations_by_path=evaluations,
    )
    write_csv(main_ci_rows, root / "artifacts/tables/main_architecture_bootstrap_ci.csv")
    write_markdown(main_ci_rows, root / "artifacts/tables/main_architecture_bootstrap_ci.md")

    model_ci_rows = evaluate_model_comparison_bootstrap(
        manifest,
        evaluations_by_path=evaluations,
    )
    write_csv(model_ci_rows, root / "artifacts/tables/model_comparison_bootstrap_ci.csv")
    write_markdown(model_ci_rows, root / "artifacts/tables/model_comparison_bootstrap_ci.md")

    cross_family_ci_rows = evaluate_cross_family_bootstrap(
        manifest,
        evaluations_by_path=evaluations,
    )
    write_csv(cross_family_ci_rows, root / "artifacts/tables/cross_family_bootstrap_ci.csv")
    write_markdown(cross_family_ci_rows, root / "artifacts/tables/cross_family_bootstrap_ci.md")

    print(f"Wrote {len(rows)} table row(s) to artifacts/tables/main_results.*")
    print(
        f"Wrote {len(main_ci_rows)} bootstrap CI row(s) to "
        "artifacts/tables/main_architecture_bootstrap_ci.*"
    )
    print(
        f"Wrote {len(model_ci_rows)} bootstrap CI row(s) to "
        "artifacts/tables/model_comparison_bootstrap_ci.*"
    )
    print(
        f"Wrote {len(cross_family_ci_rows)} bootstrap CI row(s) to "
        "artifacts/tables/cross_family_bootstrap_ci.*"
    )


if __name__ == "__main__":
    main()
