# Table Artifacts

Generated paper tables are written here by `scripts/reproduce_tables.py`.

`main_results.csv` is the canonical no-API reproduction table. It includes the
primary micro-averaged metrics, secondary macro-averaged metrics, denominator
counts, and additional recall@k fields used by the appendix tables.
Primary metrics are GO semantic recall@30, anatomy semantic recall@10, and
synonym exact recall@20.

`*_bootstrap_ci.csv` files report 95% percentile bootstrap confidence intervals
for the primary paper comparisons: main architecture scaling, fixed-harness
model comparison, and cross-family harness comparison. The bootstrap resamples
genes with replacement and recomputes the same micro-averaged recall estimators
used in `main_results.csv`.
