# Offshore wave distribution adaptation

Version 0.2.0, 18 October 2026. Author: Yujie Chen.

Repository: https://github.com/chen5512008/myPaper

This package provides the companion code, fixed evaluation tables and reproducibility materials. The scientific task is 6, 12, 24 and 48 hour significant wave height quantile forecasting from the previous 168 hours at six Chinese offshore ERA5 points. Reference distributions are adapted with median and ordered gap corrections. Reference-specific ablations assess query information; weather masks define explicit fallback. A separate complete-input comparison evaluates TabM, boosted trees and a frozen state aggregation. Most of its gain over enhanced LightGBM is already explained by TabM; the additional aggregation gain is small.

## Environment and figure reproduction

Use Python 3.12.14 and install requirements.txt in a virtual environment. The recorded CUDA environment used torch 2.9.1+cu128 on a 4 GB RTX 3050 Ti; select the appropriate official PyTorch wheel for the target platform. At least 16 GB RAM is recommended. Run GPU experiments sequentially. CatBoost GPU retraining is not guaranteed bitwise reproducible. Source code is MIT licensed; data and external software retain their licences.

    python -m pip install -r requirements.txt
    python reproduce.py --stage figures

The figure stage is self-contained in the source archive: it reads provided result tables and fixed gate summaries, requires no NetCDF download, and generates the new ablation/interruption and aggregation diagnostic figures. Fonts affect pixels; Microsoft YaHei was used for Chinese labels in the recorded Windows environment. Other environments may substitute a Chinese font. The complete manuscript's older figures have their source scripts in research; their full data/checkpoints are separate assets.

## Review experiments and fixed model replay

Extract reproduction-assets.zip into this repository root, preserving the research/ directory. Then:

    python reproduce.py --stage review-metrics
    python reproduce.py --stage review-trees
    python reproduce.py --stage review-adapters
    python reproduce.py --stage review-training
    python reproduce.py --stage fusion-metrics

Metrics recomputes all empirical ablation seeds and common-mask diagnostics. The fusion-metrics stage replays the frozen gate from all seven cached expert forecasts, then recomputes three-period scores and synchronized block intervals. It does not claim to regenerate the cached foundation-model forecasts. The bundled gate resolves its original recorded path to the matching local selection checkpoint, without modifying any weights. Prediction/training scripts skip completed outputs; to retrain, use a fresh working copy without the generated run folders. The review training stage runs only the requested new seeds 29 and 43 for the three empirical ablations, with the original recipe and 2020–2021 early stopping. Existing full-reference and seed 17 ablation checkpoints/predictions are supplied. The scripts' relative paths remain portable. Model/prediction hashes and masks are archived. Inputs after an origin never enter features.

The source tree also retains the earlier full experiments and modern-expert training code. Full foundation-model retraining/inference requires the corresponding licensed external weights and the source snapshots listed in DATA.md; the figure and review stages do not claim to retrain those backbones. Some archived research entry points contain original local-path assumptions; reproduce.py is the portable entry point for the stages above. Foundation checkpoints are not relicensed or bundled. Package versions and original protocol records are retained.

## Evaluation status

2000–2019: training. 2020–2021: original model selection. 2020: new-expert early stopping and aggregation fitting; 2021: aggregation selection. 2022–2025 and 2026H1 were already inspected. 2026Q3 was reserved before aggregation selection, but its scores are now known and it is not an untouched set for subsequent development. The new reviewer analyses are exploratory robustness checks with fixed recipes, not score-driven optimization. Chronos ablations and spatial folds retain one seed. Empirical full and three empirical ablations use seeds 17, 29 and 43. Seven/28 day circular blocks keep all sites synchronized. High-wave macro and pooled metrics are explicitly distinguished.

Weather interruptions compare frozen complete-trained enhanced LightGBM under native NaN and fixed training-mean policies with mask-trained adapters. This does not isolate a training-augmentation-independent architectural robustness effect. No interrupted-input aggregation or independent buoy validation is claimed. Marginal coverage near 80% does not resolve the low high-wave conditional coverage of the base adapters.

## Snapshot assets and archival status

Source code and exported tables are separated from the larger fixed snapshot. The snapshot archive is named reproduction-assets.zip (567,090,556 bytes; SHA-256 b21e31d1a628b23626632eeeb6b88533ff4c5e54d6d84ab6d649f64f546945fa). ASSETS.json lists all 221 internal file hashes. The asset archive still requires external upload; no public asset URL or software archive DOI has been issued. The figure stage runs from the source package alone. Cached metric replay and the documented training stages additionally require the snapshot archive. Foundation checkpoint acquisition is a separate licensed dependency.

CITATION.cff and .zenodo.json contain author/version metadata. The repository URL is recorded above and in CITATION.cff. A software archive DOI is pending and will be added after archival confirmation. Machine-specific paths in provenance metadata are reduced to filenames; metadata_path_migration.json records this change. Numerical configurations and archived prediction/model bytes are preserved.

## Code map

The portable entry point is reproduce.py. research/review_followup_20261018/model.py implements the historical adapter; run_repeats.py fixes training and empirical ablation variants; diagnostics.py constructs common weather masks and compares both frozen LightGBM missing-value policies; summarize_repeats.py computes paired seed effects. Its figures.py and method_figure.py regenerate the revised figures.

research/fusion_revision_20261017/train_tabm.py, train_trees.py and train_cat_point.py train the modern experts. gate.py and fusion.py implement state and convex quantile aggregation; evaluate_repeats.py scores the frozen predictions and synchronized time blocks. Legacy experiment entry points retain their original chronological role; reproduce.py is the supported portable interface.

Run python verify_assets.py after extracting the snapshot to check all 221 file hashes. No credential files or internal reviewer correspondence are part of this source package.

The independent-directory replay record is in REPRODUCTION_CHECK.json. It confirms 30 byte-identical figure files and four recomputed result tables at numerical tolerance 1e-12 in the recorded environment.
