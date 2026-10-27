# Analysis code for *What Predicts Creative Self-Efficacy?*

This repository contains the Python analysis code accompanying the manuscript
*What Predicts Creative Self-Efficacy? A Cross-Validated Machine Learning Study*.

The participant-level EPOCH-A survey data are **not included**. They contain
potentially sensitive survey information, and the authors do not have
unrestricted authority to redistribute the source dataset. Access requests must
be considered by the corresponding author and the relevant ethics,
institutional, data-governance, and data-custodian authorities.

## Environment

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). Create the
locked environment with:

```bash
uv sync
```

`requirements.txt` is supplied as a compatibility option for users who do not
use uv.

## Local data placement

Authorized users should place the SPSS file at:

```text
data/raw/EPOCH-A 全国数字行为与环境心理横断面样本-清洗后样本.sav
```

The `data/` directory is ignored by Git. Do not commit or redistribute the
participant-level file.

## Analysis sequence

Run the principal analyses from the repository root:

```bash
uv run python scripts/construct_level_cv.py
uv run python scripts/robustness_measurement.py
uv run python scripts/construct_predictor_profile.py
uv run python scripts/unified_algorithm_benchmark.py
uv run python scripts/ordinal_measurement.py
uv run python scripts/domain_shapley.py
uv run python scripts/cross_model_importance_stability.py
uv run python scripts/self_evaluation_structure_sensitivity.py
uv run python scripts/flow_measurement_sensitivity_complete.py
uv run python scripts/build_apa_figures.py
```

The scripts write derived outputs beneath `results/`. Some intermediate files
contain participant-level out-of-fold predictions and must not be shared
publicly without the same approvals as the source data. The entire `results/`
directory is therefore ignored in this public code repository.

The teaching notebook in `notebooks/` has had all cell outputs removed. It
documents the workflow but does not contain study records.

## Analysis scope

The workflow includes repeated nested cross-validation, comparison of linear
and tree-based algorithms, domain deletion and retraining, exact domain Shapley
allocation across all 64 domain combinations, construct-level permutation
importance, ordinal measurement models, and prespecified sensitivity analyses.
The flow-measurement sensitivity workflow repeats the complete six-domain
analysis and exact 64-subset Shapley decomposition after removing the three
flow items judged closest in meaning to efficacy beliefs, and also evaluates
the corresponding random-forest deletion result.
The results are predictive associations from cross-sectional self-report data;
they are not causal estimates.

## License

The analysis code is released under the MIT License. This license applies to
the code only and does not grant access to, or permission to redistribute, the
study data or questionnaire materials.
