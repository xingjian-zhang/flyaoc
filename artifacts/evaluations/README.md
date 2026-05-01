# Evaluation Artifacts

Derived metric summaries belong here after running reproduction scripts.
These files should be reproducible from `artifacts/predictions/` plus the HF
benchmark labels.

Primary paper metrics use micro-averaged recall@k over current verified
corpus-grounded facts, with failed or empty-output prediction rows scored as
zero recall rather than excluded.
