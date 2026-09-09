"""
run_analysis.py
================
Top-level script: runs the full global kinetic analysis (see
alkb_kinetics.py) on all three AlkB dimethyl substrates (m22G, m66A, m44C)
and reproduces Table 1 and Table S10 of the manuscript.

Usage:
    python run_analysis.py

Outputs (written to outputs/):
    - table1.csv              : final rate constants and half-lives (Table 1)
    - table_s10.csv            : model-selection / goodness-of-fit statistics
    - <substrate>_fit.png       : data + global fit overlay figure per substrate
    - log.txt                   : full console output of the analysis
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alkb_kinetics import load_substrate_csv, analyze, model_extended, model_simple

SUBSTRATES = ["m22G", "m66A", "m44C"]
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def make_figure(t, A, B, C, names, result, out_path):
    A0, B0, Total = result["A0"], result["B0"], result["Total"]
    use_extended = result["use_extended"]
    params = result["fit_extended"].x if use_extended else result["fit_simple"].x
    model = model_extended if use_extended else model_simple

    tt = np.linspace(max(t.min(), 1e-3), t.max(), 400)
    Am, Bm, Cm = model(params, tt, A0, B0, Total)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [(names[0], A, Am, 0), (names[1], B, Bm, 1), (names[2], C, Cm, 2)]
    for name, raw, curve, idx in panels:
        ax = axes[idx]
        mean, sd = raw.mean(1), raw.std(1, ddof=1)
        for rep in range(raw.shape[1]):
            ax.scatter(t, raw[:, rep], color="0.5", s=18, alpha=0.6, zorder=2,
                       label="replicates" if rep == 0 else None)
        ax.errorbar(t, mean, yerr=sd, fmt="ko", ms=5, capsize=3, zorder=3, label="mean ± SD")
        ax.plot(tt, curve, color="#1f9e4c", lw=2.2, label="global fit")
        ax.set_xscale("symlog", linthresh=0.5)
        xticks = [x for x in [0, 0.5, 1, 2, 5, 10, 20, 40] if x <= t.max()]
        ax.set_xticks(xticks); ax.set_xticklabels([str(x) for x in xticks])
        ax.set_xlim(-0.05, t.max() * 1.25); ax.set_ylim(-5, 105)
        ax.set_xlabel("Time (min)"); ax.set_ylabel("% species"); ax.set_title(name)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "log.txt")
    log_f = open(log_path, "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log_f.write(s)
        def flush(self):
            sys.__stdout__.flush(); log_f.flush()
    sys.stdout = Tee()

    table1_rows = []
    table_s10_rows = []

    for substrate in SUBSTRATES:
        csv_path = os.path.join(DATA_DIR, f"{substrate}.csv")
        t, A, B, C, names = load_substrate_csv(csv_path)

        print(f"\n{'='*72}\n{substrate}  ({names[0]} -> {names[1]} -> {names[2]})\n{'='*72}")
        result = analyze(t, A, B, C, names)

        fig_path = os.path.join(OUT_DIR, f"{substrate}_fit.png")
        make_figure(t, A, B, C, names, result, fig_path)
        print(f"Saved figure: {fig_path}")

        use_ext = result["use_extended"]
        ci = result["bootstrap_ci"]
        if use_ext:
            f, k1f, k1s, k2 = result["fit_extended"].x
            pl = result["profile_likelihood_k1_fast"]
            f_ci, k1f_ci, k1s_ci, k2_ci = ci
            if pl["unresolved_upper"]:
                k1f_str = f">{pl['lower']:.2f} (bound only)"
                t_half_fast = f"<{np.log(2)/pl['lower']*60:.0f} s"
            else:
                k1f_str = f"{k1f:.3f} ({k1f_ci[0]:.3f}-{k1f_ci[1]:.3f})"
                t_half_fast = f"{np.log(2)/k1f*60:.0f} s"
            table1_rows.append(dict(
                substrate=substrate, f=f"{f:.3f} ({f_ci[0]:.3f}-{f_ci[1]:.3f})",
                k1_fast=k1f_str, k1_slow=f"{k1s:.4f} ({k1s_ci[0]:.4f}-{k1s_ci[1]:.4f})",
                k2=f"{k2:.4f} ({k2_ci[0]:.4f}-{k2_ci[1]:.4f})",
                t_half_fast=t_half_fast, t_half_slow=f"{np.log(2)/k1s:.1f} min",
                t_half_step2=f"{np.log(2)/k2:.0f} min"))
        else:
            k1, k2 = result["fit_simple"].x
            k1_ci, k2_ci = ci
            table1_rows.append(dict(
                substrate=substrate, f="n/a (single population)",
                k1_fast="n/a", k1_slow=f"{k1:.4f} ({k1_ci[0]:.4f}-{k1_ci[1]:.4f})",
                k2=f"{k2:.4f} ({k2_ci[0]:.4f}-{k2_ci[1]:.4f})",
                t_half_fast="n/a", t_half_slow=f"{np.log(2)/k1:.1f} min",
                t_half_step2=f"{np.log(2)/k2:.0f} min"))

        diag, comp, r2 = result["diagnostic"], result["comparison"], result["r2"]
        table_s10_rows.append(dict(
            substrate=substrate,
            diag_F=f"F({diag['dof'][0]},{diag['dof'][1]})={diag['F']:.1f}",
            diag_p=f"{diag['p_value']:.2e}",
            comp_F=f"F({comp['dof'][0]},{comp['dof'][1]})={comp['F']:.1f}",
            comp_p=f"{comp['p_value']:.2e}", delta_aic=f"{comp['delta_aic']:.1f}",
            r2_substrate=f"{r2['substrate']:.3f}", r2_intermediate=f"{r2['intermediate']:.3f}",
            r2_product=f"{r2['product']:.3f}", r2_global=f"{r2['global_']:.3f}"))

    # ---- write Table 1 ----
    t1_path = os.path.join(OUT_DIR, "table1.csv")
    with open(t1_path, "w") as fh:
        cols = ["substrate", "f", "k1_fast", "k1_slow", "k2", "t_half_fast", "t_half_slow", "t_half_step2"]
        fh.write(",".join(cols) + "\n")
        for row in table1_rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"\nSaved: {t1_path}")

    # ---- write Table S10 ----
    ts_path = os.path.join(OUT_DIR, "table_s10.csv")
    with open(ts_path, "w") as fh:
        cols = ["substrate", "diag_F", "diag_p", "comp_F", "comp_p", "delta_aic",
                "r2_substrate", "r2_intermediate", "r2_product", "r2_global"]
        fh.write(",".join(cols) + "\n")
        for row in table_s10_rows:
            fh.write(",".join(str(row[c]) for c in cols) + "\n")
    print(f"Saved: {ts_path}")

    sys.stdout = sys.__stdout__
    log_f.close()
    print(f"\nDone. All outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
