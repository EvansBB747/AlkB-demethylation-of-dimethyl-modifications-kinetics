# AlkB sequential demethylation kinetics

Code and data for the global kinetic analysis of AlkB-catalyzed oxidative
demethylation of exocyclic dimethyl DNA/RNA base modifications (m22G, m66A,
m44C), reported in Table 1 and Table S10 of the accompanying manuscript.

## What this does

For each dimethyl substrate, AlkB converts:

```
dimethyl substrate --k1--> monomethyl intermediate --k2--> unmodified base
      m22G -> m2G -> G
      m66A -> m6A -> A
      m44C -> m4C -> C
```

Diagnostic analysis showed that substrate depletion is not adequately
described by a single first-order rate constant for any of the three
substrates (extra-sum-of-squares F-test comparing mono- and
bi-exponential decay, p<10⁻⁷ in every case). Data were therefore fit to
an extended model in which a fraction `f` of the dimethyl substrate
reacts with rate constant `k1_fast` and the remaining fraction `(1-f)`
reacts with `k1_slow`, both converging on a shared monomethyl
intermediate pool that is converted to the unmodified base with rate
constant `k2`. This four-parameter model was compared against the
corresponding single-population model by extra-sum-of-squares F-test and
AIC for each substrate (see Methods).

The pipeline, run independently per substrate:

1. **Diagnostic F-test** — mono- vs. bi-exponential decay of the substrate
   channel alone (`mono_vs_bi_exponential_test`).
2. **Global weighted nonlinear least-squares fit** of both the simple
   (single-population) and extended (two-population) models,
   simultaneously across all three species (substrate, intermediate,
   product) and all replicates, with weights from inverse replicate
   variance (`fit_simple_model`, `fit_extended_model`).
3. **Model selection** between the two by extra-sum-of-squares F-test and
   ΔAIC (`compare_models`).
4. **Robustness check**: refitting from 300 randomized starting points to
   confirm convergence to a global optimum (`multistart_check`).
5. **Uncertainty quantification**: case-resampling bootstrap (N=2000,
   `bootstrap_ci`) and, for `k1_fast`, a 1D profile-likelihood scan
   (`profile_likelihood_k1_fast`) — used because `k1_fast` for m66A is not
   identifiable from the current time course (the fast-reacting fraction
   reacts to near-completion within the earliest sampled time point) and
   is reported only as a one-sided 95% lower bound rather than a point
   estimate.
6. **Goodness of fit**: R² per species and globally, from the selected
   model.

## Repository layout

```
alkb_kinetics/
├── README.md              <- this file
├── requirements.txt        <- pinned package versions
├── alkb_kinetics.py         <- analysis functions (the library)
├── run_analysis.py          <- runs the full pipeline on all 3 substrates
├── data/
│   ├── m22G.csv
│   ├── m66A.csv
│   └── m44C.csv
└── outputs/                 <- created by run_analysis.py
    ├── table1.csv
    ├── table_s10.csv
    ├── <substrate>_fit.png
    └── log.txt
```

## Data format

Each `data/<substrate>.csv` file holds the time course in wide format:
one row per time point, three replicate columns each for the dimethyl
substrate, monomethyl intermediate, and unmodified base (all values are
LC-MS-derived concentrations, expressed as % of total signal, corrected
using species-specific calibration curves — see Methods). Column names
follow `<species>_rep1/2/3`.

## Running the analysis

```bash
pip install -r requirements.txt
python run_analysis.py
```

This reproduces the fitted rate constants, model-selection statistics,
and goodness-of-fit values reported in Table 1 and Table S10, and writes
per-substrate figures (data + global fit overlay) to `outputs/`. Console
output (including the full multi-start and profile-likelihood diagnostics)
is also saved to `outputs/log.txt`.

Runtime is a few minutes total (dominated by the 2000-iteration bootstrap
and 300-start robustness check per substrate); reduce `n_boot` /
`n_multistart` in `analyze()` for a faster, lower-precision run.

## Notes on reproducibility

- All fits use `scipy.optimize.least_squares` with the trust-region-
  reflective algorithm and explicit parameter bounds (see
  `alkb_kinetics.py` docstrings and the Methods section).
- A reported p-value of `0.00e+00` in `table_s10.csv` reflects
  floating-point underflow (the true p-value is astronomically small,
  e.g. <10⁻¹⁵), not exactly zero.
- Random seeds are fixed (`seed=42`) for the multi-start and bootstrap
  routines, so results are exactly reproducible on repeated runs with the
  same package versions.

## Citation

If you use this code, please cite the accompanying manuscript [citation
to be added upon publication].
