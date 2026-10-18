# PISA 2022 creativity-resource functions

Code and aggregate results for six resource-pair comparisons of creative-thinking performance: openness–mathematics, efficacy–school, openness–perseverance, efficacy–ESCS, school–peers/family, and openness–school.

## Design and interpretation

The analysis uses 499,843 matched creative-thinking records in the public-use files (63 systems). Pair-specific complete-case samples span 41–62 systems. Cyprus is suppressed from public-use microdata; Costa Rica has no nonmissing ESCS values in these files. Ten plausible values and 80 Fay replicate weights propagate assessment and survey uncertainty in the regression models. System coefficients receive equal weight. Six principal quadratic products receive Holm correction. An earlier seven-system screen formed part of a broader 50-test family; the present analysis is exploratory.

School prediction uses three seeds, three outer school folds and two inner school folds. Transfer uses five outer system folds, two inner system folds and at most 1,000 source records per system. Target covariate distributions are available for scaling; target outcomes enter only evaluation. The transfer-loss decomposition was added after the validation results, with fitted settings held fixed. These cross-sectional comparisons estimate associations and predictive increments.

## Check the released aggregates

The recorded environment is Python 3.11.10. Install the packages in requirements.txt in a virtual environment, then run:

    python verify_aggregates.py

This recomputes the six principal system averages, Holm-adjusted probabilities, held-school increments and transfer-loss identities from the released aggregate files. No student data are needed for this check. MANIFEST.json records SHA-256 hashes of release files.

## Reproduce from OECD data

Download and extract CY08MSP_STU_QQQ.SAV and CY08MSP_CRT_COG.SAV from the [OECD PISA 2022 database](https://www.oecd.org/en/data/datasets/pisa-2022-database.html). Set the PISA2022_RAW environment variable to their containing directory. Source hashes and item identifiers are in source_metadata.json. The scoring is fixed to the original analysis: ST334/ST340/ST307 have ten items each; ST335/ST336 have six each, with nonconsecutive identifiers; ST307 items 04/06/07/10 are reversed.

Use a new checkout or copy for recomputation. The following command requires an empty exp directory, preserving the released snapshot in the original checkout:

    python reproduce.py --run --workers 4

Move the supplied exp directory outside the new checkout before that command. It rebuilds local student extracts under data/derived, then fits regression, school prediction, transfer, decomposition and simulation stages. This is a substantial CPU computation. Repeated seeds reuse the same cohort. Full regression sensitivities and all reported failures or null increments remain in the released results.

## Contents and data access

The repository contains statistical code, system/fold/simulation summaries and source/environment metadata. It contains no student records, derived person-level extracts, training-selection files, manuscript, publisher full texts or private project logs. Public-use source data remain subject to OECD access and reuse conditions. This repository does not redistribute them. Statistical source files retain their original variable identifiers to support reproduction.
