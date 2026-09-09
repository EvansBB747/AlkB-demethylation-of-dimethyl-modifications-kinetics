"""
alkb_kinetics.py
=================
Global sequential-kinetics analysis of AlkB-catalyzed oxidative demethylation
of exocyclic dimethyl DNA/RNA base modifications (m22G, m66A, m44C), as
described in the Methods section of the accompanying manuscript.

For each dimethyl substrate, AlkB converts:
    dimethyl substrate --k1--> monomethyl intermediate --k2--> unmodified base
    (m22G -> m2G -> G;  m66A -> m6A -> A;  m44C -> m4C -> C)

Diagnostic analysis of the substrate-depletion time course showed that a
single first-order process (a single k1) is statistically inadequate for
all three substrates (see `mono_vs_bi_exponential_test`). Data were
therefore fit to an extended model in which a fraction f of substrate
reacts with rate constant k1_fast and the remaining fraction (1-f) reacts
with rate constant k1_slow, both converging on a shared monomethyl
intermediate pool that is converted to the unmodified base with rate
constant k2:

    A0*f     --k1_fast--> \\
                            }--k2--> C   (shared intermediate/product pool)
    A0*(1-f) --k1_slow--> /

This module provides:
    - mono_vs_bi_exponential_test : diagnostic F-test on substrate decay
    - fit_simple_model / fit_extended_model : global weighted NLS fits
    - compare_models : extra-sum-of-squares F-test + AIC model selection
    - multistart_check : robustness check against local optima
    - bootstrap_ci : case-resampling confidence intervals
    - profile_likelihood_k1_fast : 1D profile-likelihood CI/bound for k1_fast
    - analyze : runs the full pipeline for one substrate end to end
    - load_substrate_csv : reads a data/<substrate>.csv file into arrays

See run_analysis.py for the top-level script that runs this on all three
substrates and reproduces Table 1 / Table S10 of the manuscript.

Requires: numpy, scipy, matplotlib (see requirements.txt for pinned versions)
"""
import csv
import numpy as np
from scipy.optimize import least_squares, curve_fit
from scipy import stats


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_substrate_csv(path):
    """
    Load a data/<substrate>.csv file (wide format: time_min, then 3 replicate
    columns each for substrate, intermediate, and product) into time array
    and three (n_timepoints x 3) replicate arrays.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    data = np.array(rows, dtype=float)
    t = data[:, 0]
    A = data[:, 1:4]
    B = data[:, 4:7]
    C = data[:, 7:10]
    names = tuple(h.split("_rep1")[0] for h in header[1:10:3])
    return t, A, B, C, names


# --------------------------------------------------------------------------
# Kinetic models
# --------------------------------------------------------------------------
def model_simple(params, tt, A0, B0, Total):
    """Single-population sequential model: A --k1--> B --k2--> C."""
    k1, k2 = params
    Am = A0 * np.exp(-k1 * tt)
    d = k2 - k1 if abs(k2 - k1) > 1e-9 else 1e-9
    Bm = (k1 * A0 / d) * (np.exp(-k1 * tt) - np.exp(-k2 * tt)) + B0 * np.exp(-k2 * tt)
    Cm = Total - Am - Bm
    return Am, Bm, Cm


def model_extended(params, tt, A0, B0, Total):
    """
    Two-population sequential model: fraction f of A reacts with k1_fast,
    fraction (1-f) reacts with k1_slow; both feed a shared B pool that is
    converted to C with rate constant k2.
    """
    f, k1a, k1b, k2 = params
    A1 = A0 * f * np.exp(-k1a * tt)
    A2 = A0 * (1 - f) * np.exp(-k1b * tt)
    Am = A1 + A2
    d1 = k2 - k1a if abs(k2 - k1a) > 1e-9 else 1e-9
    d2 = k2 - k1b if abs(k2 - k1b) > 1e-9 else 1e-9
    Bm = (k1a * A0 * f / d1) * (np.exp(-k1a * tt) - np.exp(-k2 * tt)) + \
         (k1b * A0 * (1 - f) / d2) * (np.exp(-k1b * tt) - np.exp(-k2 * tt)) + \
         B0 * np.exp(-k2 * tt)
    Cm = Total - Am - Bm
    return Am, Bm, Cm


# --------------------------------------------------------------------------
# Weighting
# --------------------------------------------------------------------------
def replicate_weights(A, B, C):
    """
    Inverse-replicate-variance weights, floored at half the pooled nonzero
    replicate standard deviation (avoids singular weights at time points
    with zero observed variance).
    """
    n_rep = A.shape[1]
    sds = np.concatenate([A.std(1, ddof=1), B.std(1, ddof=1), C.std(1, ddof=1)])
    pooled = sds[sds > 0].mean() if (sds > 0).any() else 1.0
    wA = np.repeat(np.maximum(A.std(1, ddof=1), pooled * 0.5), n_rep)
    wB = np.repeat(np.maximum(B.std(1, ddof=1), pooled * 0.5), n_rep)
    wC = np.repeat(np.maximum(C.std(1, ddof=1), pooled * 0.5), n_rep)
    return wA, wB, wC


# --------------------------------------------------------------------------
# Diagnostic: mono- vs bi-exponential substrate decay
# --------------------------------------------------------------------------
def mono_vs_bi_exponential_test(t, A):
    """
    Extra-sum-of-squares F-test comparing mono- and bi-exponential decay of
    the substrate (A) time course alone. Returns a dict with both fits and
    the F-test result. A significant result (p<0.05) indicates the
    substrate population is not kinetically homogeneous.
    """
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af = A.flatten()
    A0 = A[0].mean()

    def mono(tt, a0, k):
        return a0 * np.exp(-k * tt)

    def bi(tt, a0, f, kf, ks):
        return a0 * (f * np.exp(-kf * tt) + (1 - f) * np.exp(-ks * tt))

    p1, _ = curve_fit(mono, t_rep, Af, p0=[A0, 0.3], bounds=([50, 1e-5], [110, 20]))
    p2, _ = curve_fit(bi, t_rep, Af, p0=[A0, 0.3, 3, 0.05],
                       bounds=([50, 0, 1e-5, 1e-6], [110, 1, 50, 5]))
    ssr1 = np.sum((Af - mono(t_rep, *p1)) ** 2)
    ssr2 = np.sum((Af - bi(t_rep, *p2)) ** 2)
    n = len(Af)
    df1, df2 = n - 2, n - 4
    F = ((ssr1 - ssr2) / (df1 - df2)) / (ssr2 / df2)
    p = 1 - stats.f.cdf(F, df1 - df2, df2)
    return dict(mono_params=p1, bi_params=p2, ssr_mono=ssr1, ssr_bi=ssr2,
                F=F, dof=(df1 - df2, df2), p_value=p)


# --------------------------------------------------------------------------
# Global fits
# --------------------------------------------------------------------------
def _resid(model, params, t_rep, A0, B0, Total, Af, Bf, Cf, wA, wB, wC):
    Am, Bm, Cm = model(params, t_rep, A0, B0, Total)
    return np.concatenate([(Am - Af) / wA, (Bm - Bf) / wB, (Cm - Cf) / wC])


def fit_simple_model(t, A, B, C, p0=(0.15, 0.01)):
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af, Bf, Cf = A.flatten(), B.flatten(), C.flatten()
    A0, B0, Total = A[0].mean(), B[0].mean(), (A + B + C).mean()
    wA, wB, wC = replicate_weights(A, B, C)
    fit = least_squares(
        lambda p: _resid(model_simple, p, t_rep, A0, B0, Total, Af, Bf, Cf, wA, wB, wC),
        p0, bounds=([1e-5, 1e-6], [20, 5]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return fit, (A0, B0, Total, wA, wB, wC)


def fit_extended_model(t, A, B, C, p0=(0.3, 1.0, 0.05, 0.01)):
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af, Bf, Cf = A.flatten(), B.flatten(), C.flatten()
    A0, B0, Total = A[0].mean(), B[0].mean(), (A + B + C).mean()
    wA, wB, wC = replicate_weights(A, B, C)
    fit = least_squares(
        lambda p: _resid(model_extended, p, t_rep, A0, B0, Total, Af, Bf, Cf, wA, wB, wC),
        p0, bounds=([0.001, 1e-3, 1e-5, 1e-5], [0.999, 50, 5, 5]),
        xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return fit, (A0, B0, Total, wA, wB, wC)


def compare_models(fit_simple, fit_extended, n_params_simple=2, n_params_extended=4):
    """Extra-sum-of-squares F-test + AIC comparing the simple and extended models."""
    ssr_s = fit_simple.fun @ fit_simple.fun
    ssr_e = fit_extended.fun @ fit_extended.fun
    n = len(fit_simple.fun)
    dof_s, dof_e = n - n_params_simple, n - n_params_extended
    F = ((ssr_s - ssr_e) / (dof_s - dof_e)) / (ssr_e / dof_e)
    p = 1 - stats.f.cdf(F, dof_s - dof_e, dof_e)
    aic_s = n * np.log(ssr_s / n) + 2 * n_params_simple
    aic_e = n * np.log(ssr_e / n) + 2 * n_params_extended
    return dict(ssr_simple=ssr_s, ssr_extended=ssr_e, F=F, dof=(dof_s - dof_e, dof_e),
                p_value=p, aic_simple=aic_s, aic_extended=aic_e, delta_aic=aic_s - aic_e)


def multistart_check(t, A, B, C, model="extended", n_starts=300, seed=42):
    """Refit from many randomized starting points to check for a global optimum."""
    rng = np.random.default_rng(seed)
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af, Bf, Cf = A.flatten(), B.flatten(), C.flatten()
    A0, B0, Total = A[0].mean(), B[0].mean(), (A + B + C).mean()
    wA, wB, wC = replicate_weights(A, B, C)

    if model == "extended":
        mfun = model_extended
        bounds = ([0.001, 1e-3, 1e-5, 1e-5], [0.999, 50, 5, 5])
        starts = np.column_stack([
            rng.uniform(0.05, 0.95, n_starts),
            rng.uniform(0.1, 20, n_starts),
            rng.uniform(0.001, 1, n_starts),
            rng.uniform(0.0005, 0.05, n_starts),
        ])
    else:
        mfun = model_simple
        bounds = ([1e-5, 1e-6], [20, 5])
        starts = np.column_stack([
            rng.uniform(0.01, 5, n_starts),
            rng.uniform(0.0005, 0.5, n_starts),
        ])

    costs = []
    for s0 in starts:
        try:
            r = least_squares(lambda p: _resid(mfun, p, t_rep, A0, B0, Total, Af, Bf, Cf, wA, wB, wC),
                               s0, bounds=bounds, xtol=1e-13, ftol=1e-13, gtol=1e-13)
            costs.append(r.cost)
        except Exception:
            pass
    costs = np.array(costs)
    frac_at_best = float(np.mean(np.isclose(costs, costs.min(), rtol=1e-3))) if len(costs) else np.nan
    return dict(n_starts_tried=len(costs), best_cost=float(costs.min()) if len(costs) else np.nan,
                frac_converged_to_best=frac_at_best)


def bootstrap_ci(t, A, B, C, best_params, model="extended", n_boot=2000, seed=42):
    """Case-resampling bootstrap (resample replicates with replacement) for parameter CIs."""
    rng = np.random.default_rng(seed)
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    A0, B0, Total = A[0].mean(), B[0].mean(), (A + B + C).mean()
    wA, wB, wC = replicate_weights(A, B, C)
    mfun = model_extended if model == "extended" else model_simple
    bounds = ([0.001, 1e-3, 1e-5, 1e-5], [0.999, 50, 5, 5]) if model == "extended" else ([1e-5, 1e-6], [20, 5])

    boots = []
    for _ in range(n_boot):
        iA = rng.integers(0, n_rep, n_rep)
        iB = rng.integers(0, n_rep, n_rep)
        iC = rng.integers(0, n_rep, n_rep)
        Ab, Bb, Cb = A[:, iA].flatten(), B[:, iB].flatten(), C[:, iC].flatten()
        try:
            r = least_squares(
                lambda p: _resid(mfun, p, t_rep, A0, B0, Total, Ab, Bb, Cb, wA, wB, wC),
                best_params, bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            boots.append(r.x)
        except Exception:
            pass
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5], axis=0).T if len(boots) else None
    return boots, ci


def profile_likelihood_k1_fast(t, A, B, C, best_fit_extended, grid=None):
    """
    1D profile-likelihood scan over k1_fast: fix k1_fast at each grid value,
    refit (f, k1_slow, k2), and find the 95% confidence region using the
    F-distribution threshold for one constrained parameter. If the SSR
    plateaus at the upper end of the grid (as for a fast phase that
    completes within the dead time / earliest time point), only a lower
    bound is meaningful and the returned upper bound should be treated as
    unresolved rather than a true confidence limit.
    """
    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af, Bf, Cf = A.flatten(), B.flatten(), C.flatten()
    A0, B0, Total = A[0].mean(), B[0].mean(), (A + B + C).mean()
    wA, wB, wC = replicate_weights(A, B, C)

    ssr_min = best_fit_extended.fun @ best_fit_extended.fun
    dof_min = len(best_fit_extended.fun) - 4

    if grid is None:
        grid = np.concatenate([np.arange(0.1, 2, 0.05), np.arange(2, 10, 0.2), np.arange(10, 50, 2)])

    ssrs = []
    for k1a_fixed in grid:
        def resid_fixed(p, k1a_fixed=k1a_fixed):
            f, k1b, k2 = p
            Am, Bm, Cm = model_extended([f, k1a_fixed, k1b, k2], t_rep, A0, B0, Total)
            return np.concatenate([(Am - Af) / wA, (Bm - Bf) / wB, (Cm - Cf) / wC])
        r = least_squares(resid_fixed, [0.2, 0.02, 0.005],
                           bounds=([0.001, 1e-5, 1e-5], [0.999, 5, 5]),
                           xtol=1e-13, ftol=1e-13, gtol=1e-13)
        ssrs.append(r.fun @ r.fun)
    ssrs = np.array(ssrs)

    Fcrit = stats.f.ppf(0.95, 1, dof_min)
    threshold = ssr_min * (1 + Fcrit / dof_min)
    in_ci = grid[ssrs <= threshold]
    lower = float(in_ci.min()) if len(in_ci) else np.nan
    upper = float(in_ci.max()) if len(in_ci) else np.nan
    plateau = bool(upper >= grid.max() * 0.9)  # SSR never rose back above threshold -> unresolved upper bound
    return dict(grid=grid, ssr=ssrs, threshold=threshold, lower=lower, upper=upper,
                unresolved_upper=plateau)


# --------------------------------------------------------------------------
# Full pipeline for one substrate
# --------------------------------------------------------------------------
def analyze(t, A, B, C, names, n_multistart=300, n_boot=2000, seed=42, verbose=True):
    """Run the complete analysis pipeline (diagnostic -> fit -> select -> validate) for one substrate."""
    out = {"names": names}

    diag = mono_vs_bi_exponential_test(t, A)
    out["diagnostic"] = diag
    if verbose:
        print(f"[1] Substrate mono- vs bi-exponential F-test: "
              f"F({diag['dof'][0]},{diag['dof'][1]})={diag['F']:.2f}  p={diag['p_value']:.2e}")

    fit_s, _ = fit_simple_model(t, A, B, C)
    fit_e, (A0, B0, Total, wA, wB, wC) = fit_extended_model(t, A, B, C)
    comp = compare_models(fit_s, fit_e)
    out["fit_simple"] = fit_s
    out["fit_extended"] = fit_e
    out["comparison"] = comp
    out["A0"], out["B0"], out["Total"] = A0, B0, Total
    if verbose:
        print(f"[2] Model comparison: F({comp['dof'][0]},{comp['dof'][1]})={comp['F']:.2f}  "
              f"p={comp['p_value']:.2e}  dAIC={comp['delta_aic']:.1f}")

    use_extended = (comp["p_value"] < 0.05) and (comp["delta_aic"] > 0)
    out["use_extended"] = use_extended

    ms = multistart_check(t, A, B, C, model="extended" if use_extended else "simple",
                           n_starts=n_multistart, seed=seed)
    out["multistart"] = ms
    if verbose:
        print(f"[3] Multi-start: {ms['n_starts_tried']} starts, "
              f"{ms['frac_converged_to_best']*100:.0f}% converged to best cost")

    best_params = fit_e.x if use_extended else fit_s.x
    boots, ci = bootstrap_ci(t, A, B, C, best_params,
                              model="extended" if use_extended else "simple",
                              n_boot=n_boot, seed=seed)
    out["bootstrap"] = boots
    out["bootstrap_ci"] = ci

    if use_extended:
        pl = profile_likelihood_k1_fast(t, A, B, C, fit_e)
        out["profile_likelihood_k1_fast"] = pl
        if verbose:
            tag = " (unresolved upper bound -- report as lower bound only)" if pl["unresolved_upper"] else ""
            print(f"[4] Profile-likelihood k1_fast 95% CI: ({pl['lower']:.3f}, {pl['upper']:.3f}) /min{tag}")

    def r2(y, yhat):
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    n_rep = A.shape[1]
    t_rep = np.repeat(t, n_rep)
    Af, Bf, Cf = A.flatten(), B.flatten(), C.flatten()
    model = model_extended if use_extended else model_simple
    Am, Bm, Cm = model(best_params, t_rep, A0, B0, Total)
    out["r2"] = dict(
        substrate=r2(Af, Am), intermediate=r2(Bf, Bm), product=r2(Cf, Cm),
        global_=r2(np.concatenate([Af, Bf, Cf]), np.concatenate([Am, Bm, Cm])))
    if verbose:
        print(f"[5] R2: {names[0]}={out['r2']['substrate']:.3f}  {names[1]}={out['r2']['intermediate']:.3f}  "
              f"{names[2]}={out['r2']['product']:.3f}  global={out['r2']['global_']:.3f}")

    return out
