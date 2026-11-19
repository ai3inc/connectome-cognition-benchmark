"""Figures for the manuscript, drawn from the numbers index (called by build_numbers.py --figs).

Colours (validated with the dataviz palette checker, light mode): benchmark's own models
orange #eb6834, ours blue #2a78d6; hollow markers mark the re-run / unfiltered variants so
identity never rests on colour alone. Text stays black or grey.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, GREY, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "DejaVu Serif"],
                     "svg.fonttype": "none", "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": GREY, "ytick.color": GREY, "axes.linewidth": 0.8})


def _clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1(out, V):
    rows = [  # label, value, sd, colour, filled
        ("Published: motor fMRI (KRR)", V["pub_motor"], None, ORANGE, True),
        ("Published: gambling fMRI (KRR)", V["pub_gamb"], None, ORANGE, True),
        ("Published: social fMRI (KRR)", V["pub_social"], None, ORANGE, True),
        ("Published: resting fMRI (KRR)", V["pub_rs"], None, ORANGE, True),
        ("Published: working memory fMRI (KRR)", V["pub_wm"], None, ORANGE, True),
        ("Published: language fMRI (KRR)", V["pub_lang"], None, ORANGE, True),
        ("Published: multi-kernel KRR, six fMRI", V["pub_multikrr"], None, ORANGE, True),
        ("Published: stacking, all modalities", V["pub_stackall"], None, ORANGE, True),
        ("Published: stacking, six fMRI (best)", V["pub_stack6"], V["pub_stack6_sd"], ORANGE, True),
        ("Benchmark kernel, six fMRI, our re-run", V["cbig6_r"], V["cbig6_sd"], ORANGE, False),
        ("Ours without the kernel filter", V["nofilt_r"], V["nofilt_sd"], BLUE, False),
        ("Ours, fixed settings", V["canon_r"], V["canon_sd"], BLUE, True),
        ("Ours, settings chosen inside each fold", V["nested_r"], V["nested_sd"], BLUE, True),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for i, (lab, v, sd, c, filled) in enumerate(rows):
        if sd:
            ax.errorbar(v, i, xerr=sd, color=c, lw=1.2, capsize=0, zorder=2)
        ax.plot(v, i, "o", ms=6.5, mec=c, mfc=c if filled else "white", mew=1.6, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.axvline(V["pub_stack6"], color=ORANGE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel("Prediction accuracy, mean of per-fold Pearson r (60 splits × 10 folds)")
    ax.set_xlim(0.42, 0.75)
    _clean(ax)
    h = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=ORANGE, mec=ORANGE, label="benchmark's own models (published)"),
         plt.Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=ORANGE, mew=1.6, label="benchmark kernel, re-run here"),
         plt.Line2D([], [], marker="o", ls="", ms=6, mfc=BLUE, mec=BLUE, label="ours")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2(out, V):
    steps = [("Benchmark\nkernel", V["ra_cbig6_r"], None, None),
             ("Our kernel,\nfull correlation", V["ra_fullw1_r"], V["ra_fullw1_vs_cbig6_d"], V["ra_fullw1_vs_cbig6_t"]),
             ("+ shrinkage and\npartial correlation", V["ra_partna_w0_r"], V["ra_partial_without_power_d"], V["ra_partial_without_power_t"]),
             ("+ eigenvalue\npower", V["ra_partw0_r"], V["ra_power_step_d"], V["ra_power_step_t"]),
             ("+ 5% full-\ncorrelation mix", V["ra_nofilt_r"], V["ra_mix_step_d"], V["ra_mix_step_t"]),
             ("+ kernel filter\n(all subjects)", V["ra_canon_r"], V["ra_filter_on_conditioned_d"], V["ra_filter_on_conditioned_t"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    base = 0.655
    prev = None
    for i, (lab, v, d, t) in enumerate(steps):
        if prev is None:
            ax.bar(i, v - base, bottom=base, width=0.56, color="white", edgecolor=ORANGE, lw=1.4)
        else:
            lo, hi = min(prev, v), max(prev, v)
            ax.bar(i, hi - lo, bottom=lo, width=0.56, color=BLUE if d >= 0 else "white", edgecolor=BLUE, lw=1.2)
            ax.plot([i - 1 + 0.28, i - 0.28], [prev, prev], color=GREY, lw=0.8)
            ax.text(i, hi + 0.0012, f"{d:+.4f}\nt = {t:.2f}".replace("-", "\u2212"), ha="center", va="bottom", fontsize=7.5, color=INK)
        prev = v
    xi = len(steps) - 0.62
    ax.plot([xi], [V["ra_canon_ind_r"]], marker="D", ms=6, mfc="white", mec=BLUE, mew=1.4, ls="")
    ax.annotate("filter from training\nsubjects only", (xi, V["ra_canon_ind_r"]), xytext=(xi - 1.55, V["ra_canon_ind_r"] - 0.0115),
                fontsize=7.5, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], color=INK, fontsize=7.8)
    ax.set_ylim(base, 0.732)
    ax.set_ylabel("Mean per-fold r")
    _clean(ax, "y")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3(out, V):
    rows = [  # label, d, lo, hi, primary
        ("D1 − F-BERT-8M (earlier registration)", V["ns_d1_vs_fbert_d"], V["ns_d1_vs_fbert_lo"], V["ns_d1_vs_fbert_hi"], False),
        ("D0 − F-BERT-8M (earlier registration)", V["ns_d0_vs_fbert_d"], V["ns_d0_vs_fbert_lo"], V["ns_d0_vs_fbert_hi"], False),
        ("Filter and transductive kernel (D1 − D0)", V["ns_d1_vs_d0_d"], V["ns_d1_vs_d0_lo"], V["ns_d1_vs_d0_hi"], False),
        ("D0 − plain kernel, training-only centring", V["ns_d0_vs_plain_i_d"], V["ns_d0_vs_plain_i_lo"], V["ns_d0_vs_plain_i_hi"], False),
        ("D1 − plain kernel, centring over all subjects", V["ns_d1_vs_plain_t_d"], V["ns_d1_vs_plain_t_lo"], V["ns_d1_vs_plain_t_hi"], False),
        ("D0 − benchmark kernel", V["ns_d0_vs_cbig6_d"], V["ns_d0_vs_cbig6_lo"], V["ns_d0_vs_cbig6_hi"], False),
        ("D1 − benchmark kernel  [primary]", V["ns_prim_d"], V["ns_prim_lo"], V["ns_prim_hi"], True),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    for i, (lab, d, lo, hi, prim) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=BLUE, lw=1.6 if prim else 1.2)
        ax.plot(d, i, "o", ms=7 if prim else 6, mec=BLUE, mfc=BLUE if prim else "white", mew=1.6)
    ax.axvline(0, color=GREY, lw=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.set_xlabel(f"Difference in r on the {V['ns_hold']} held-out subjects (95% bootstrap interval)")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def make(here, V, load):
    d = os.path.join(here, "figures")
    os.makedirs(d, exist_ok=True)
    fig1(os.path.join(d, "fig1_benchmark.svg"), V)
    fig2(os.path.join(d, "fig2_decomposition.svg"), V)
    fig3(os.path.join(d, "fig3_new_subjects.svg"), V)
    print("figures written")
