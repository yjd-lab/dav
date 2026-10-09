# Development instructions

- Work only within the current repository; do not create another project or worktree unless explicitly requested.
- Inspect the current Git branch, remote and working tree before editing. Yahya works on `yahya`; stop on an unexpected branch or conflicting/unrelated changes.
- Follow `docs/CONTRACT.md`, the single authoritative shared contract. Read `docs/Yahya_DAV_Project_Handoff.md` for context; its historical `docs/contracts.md` references mean `docs/CONTRACT.md`. Current user instructions control authorized scope.
- Preserve existing files, the original dataset and physical units. Keep artificial dirty demonstration data separate. Do not automatically remove meaningful sensor outliers.
- Respect ownership: Yahya owns features.py, prediction.py, advanced_analysis.py, notebook 03 and prediction tests. Amirul owns loading, validation, cleaning, transformation, statistics, main charts, scripts/run_pipeline.py and the final notebook. Do not modify teammate-owned modules without the teammate's request. Coordinate shared files and tests.
- Keep reusable Python logic in `src/industrial_analyzer/`; notebooks focus on demonstration and explanations.
- Fit model preprocessing only on training data and within training-only validation folds. Never use failure flags, identifiers or the target as prediction inputs. Use the six-feature allowlist.
- Describe this dataset as synthetic snapshot observations, not real-time readings or chronological forecasts. Never fabricate observations, predictions, metrics or test results.
- Run relevant tests after coding; report actual commands and outcomes, including failures or unavailable checks. Do not assume success.
- Never force-push, rewrite history or modify another student's branch. Do not commit or push without explicit authorization. Merge and branch changes also require authorization.
- Do not install packages, download data or begin a later implementation phase without authorization for that work. Phase 1 contains documentation, configuration and directory placeholders only.
- Resolve relative output paths from the repository root; keep generated files within it. For initial src-layout development, use PYTHONPATH=src from the root; agree on and document the environment before installation.
