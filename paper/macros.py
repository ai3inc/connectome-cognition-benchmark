"""Tables built from the numbers index, so a table cannot drift from the numbers.
Each macro returns markdown whose cells are {{key:fmt}} placeholders; resolve_and_check.py
resolves them on its next pass and counts them as used."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))


def _meta():
    return json.load(open(os.path.join(HERE, "ledger", "tables_meta.json")))


def _rows():
    return json.load(open(os.path.join(HERE, "ledger", "numbers_index.json")))["rows"]


def scoring_table(V, used):
    return "\n".join([
        "| Scoring | Ours, fixed settings | Benchmark kernel, re-run here |",
        "|---|---|---|",
        "| Pooled r over all predictions of a split, raw target | {{canon_raw_pooled:.4f}} | {{cbig6_raw_pooled:.4f}} |",
        "| Mean of per-fold r, raw target | {{canon_raw_r:.4f}} | {{cbig6_raw_r:.4f}} |",
        "| Mean of per-fold r, age band and sex regressed inside each training fold (the benchmark's scoring) | **{{canon_r:.4f}}** | **{{cbig6_r:.4f}}** |",
        "| Ours without the kernel filter, mean per-fold r, raw target, smallest ridge penalty {{lam_floor_old:sci}} | {{nofilt_raw_oldgrid:.4f}} | |",
        "| The same, smallest ridge penalty {{lam_floor_new:sci}} | {{nofilt_raw_r:.4f}} | |",
    ])


def headline_table(V, used):
    return "\n".join([
        "| Comparison (benchmark's scoring) | Difference in r | Corrected t | p |",
        "|---|---|---|---|",
        "| Ours, fixed − best published model | {{canon_vs_pub:.4f}} | — | — |",
        "| Ours, fixed − benchmark kernel re-run | {{canon_vs_cbig6_diff:.4f}} | {{canon_vs_cbig6_t:.2f}} | {{canon_vs_cbig6_p:p}} |",
        "| Ours, chosen in-fold − benchmark kernel re-run | {{nested_vs_cbig6:.4f}} | {{nested_vs_cbig6_t:.2f}} | {{nested_vs_cbig6_p:p}} |",
        "| Ours, chosen in-fold − ours, fixed | {{nested_minus_fixed:.4f}} | {{nested_minus_fixed_t:.2f}} | {{nested_minus_fixed_p:.3f}} |",
        "| Ours without the filter − benchmark kernel re-run | {{nofilt_vs_cbig6_diff:.4f}} | {{nofilt_vs_cbig6_t:.2f}} | {{nofilt_vs_cbig6_p:.3f}} |",
        "| Kernel filter (ours − ours without the filter) | {{filter_diff:.4f}} | {{filter_t:.2f}} | {{filter_p:.3f}} |",
        "| Resting state alone, ours without the filter − benchmark kernel re-run | {{rest_nofilt_vs_cbig_diff:.4f}} | {{rest_nofilt_vs_cbig_t:.2f}} | {{rest_nofilt_vs_cbig_p:sci}} |",
        "| Resting state alone, ours − benchmark kernel re-run | {{rest_canon_vs_cbig_diff:.4f}} | {{rest_canon_vs_cbig_t:.2f}} | {{rest_canon_vs_cbig_p:sci}} |",
    ])


def fm_table(V, used):
    t = lambda k: "t = {{fe_%s_t:.2f}}" % k
    return "\n".join([
        "| Setting | Subjects | D0 | D1 | Powered FC | F-BERT-8M | D0 vs F-BERT | D1 vs F-BERT |",
        "|---|---|---|---|---|---|---|---|",
        "| HCP resting state, benchmark splits | {{n_bench}} | {{fe_rest_m_d0:.4f}} | {{fe_rest_m_d1:.4f}} | {{fe_rest_m_teacher:.4f}} | {{fe_rest_m_fbert8:.4f}} | %s | %s |" % (t("rest_d0_fbert"), t("rest_d1_fbert")),
        "| HCP six modalities, benchmark splits | {{n_bench}} | {{fe_six_m_d0:.4f}} | {{fe_six_m_d1:.4f}} | {{fe_six_m_teacher:.4f}} | {{fe_six_m_fbert_cat:.4f}} | %s | %s |" % (t("six_d0_fbert"), t("six_d1_fbert")),
        "| HCP held-out subjects, six modalities | {{ns_hold}} | {{ns_d0_r:.4f}} | {{ns_d1_r:.4f}} | | {{ns_fbert_r:.4f}} | {{ns_d0_vs_fbert_lo:.3f}} to {{ns_d0_vs_fbert_hi:.3f}} | {{ns_d1_vs_fbert_lo:.3f}} to {{ns_d1_vs_fbert_hi:.3f}} |",
        "| AOMIC-ID1000, intelligence | {{aomic_n}} | {{fe_aomic_m_d0:.4f}} | {{fe_aomic_m_d1:.4f}} | {{fe_aomic_m_teacher:.4f}} | {{fe_aomic_m_fbert:.4f}} | %s | %s |" % (t("aomic_d0_fbert"), t("aomic_d1_fbert")),
        "| ABIDE-I, autism diagnosis (AUC) | {{abide_n:c}} | {{fe_abide_m_d0:.3f}} | {{fe_abide_m_d1:.3f}} | {{fe_abide_m_teacher:.3f}} | {{fe_abide_m_fbert:.3f}} | %s | %s |" % (t("abide_d0_fbert"), t("abide_d1_fbert")),
        "| ABIDE-I, full-scale IQ | {{abide_nfiq}} | {{fe_abide_fiq_m_d0:.3f}} | {{fe_abide_fiq_m_d1:.3f}} | {{fe_abide_fiq_m_teacher:.3f}} | {{fe_abide_fiq_m_fbert:.3f}} | %s | %s |" % (t("abide_fiq_d0_fbert"), t("abide_fiq_d1_fbert")),
        "| ADHD-200, diagnosis (AUC) | {{adhd_n}} | {{adhd_d0:.3f}} | {{adhd_d1:.3f}} | {{adhd_teacher:.3f}} | {{adhd_fbert:.3f}} | {{adhd_lo:.3f}} to {{adhd_hi:.3f}} | {{adhd_d1_fbert_lo:.3f}} to {{adhd_d1_fbert_hi:.3f}} |",
        "| ADHD-200, IQ | {{adhd_niq}} | {{adhd_iq_d0:.3f}} | {{adhd_iq_d1:.3f}} | {{adhd_iq_teacher:.3f}} | {{adhd_iq_fbert:.3f}} | {{adhd_iq_lo:.3f}} to {{adhd_iq_hi:.3f}} | {{adhd_iq_d1_fbert_lo:.3f}} to {{adhd_iq_d1_fbert_hi:.3f}} |",
    ])


def levers_table(V, used):
    out = ["| Change to the chain, tested after the headline was fixed | Difference in r | Corrected t |", "|---|---|---|"]
    for key, label, has_t, note in _meta()["levers"]:
        t = f"{{{{lev_{key}_t:.2f}}}}" if has_t else "—"
        lab = label[0].upper() + label[1:] + (f"; {note}" if note else "")
        out.append(f"| {lab} | {{{{lev_{key}_d:.4f}}}} | {t} |")
    return "\n".join(out)


def ext_table(V, used):
    pf = lambda k: "{{ext_%s_p:sci}}" % k if V["ext_%s_p" % k] < 0.001 else "{{ext_%s_p:.3f}}" % k
    c = lambda k: "{{ext_%s_d:.4f}} | {{ext_%s_t:.2f}} | " % (k, k) + pf(k) + " | {{ext_%s_pos}} of {{ext_reps}}" % k
    return "\n".join([
        "| Arm | Mean per-fold r |", "|---|---|",
        "| Benchmark kernel (CBIG correlation kernel) | {{ext_r_cbig:.4f}} |",
        "| Plain correlation kernel, training-only centring | {{ext_r_plain:.4f}} |",
        "| **Conditioned, training-only centring (D0-I)** | **{{ext_r_d0:.4f}}** |",
        "| Conditioned, filter from training subjects only (D1-I) | {{ext_r_d1i:.4f}} |",
        "| Conditioned, filter from all subjects (D1-T, transductive) | {{ext_r_d1t:.4f}} |",
        "",
        "| Contrast | Difference in r | Corrected t | p | Repeats positive |", "|---|---|---|---|---|",
        "| **D0-I − benchmark kernel (primary)** | " + c("d0_cbig") + " |",
        "| D0-I − plain kernel (conditioning) | " + c("d0_plain") + " |",
        "| D1-I − D0-I (filter, inductive) | " + c("d1i_d0") + " |",
        "| D1-T − D0-I (filter and all-subject centring) | " + c("d1t_d0") + " |",
        "| D1-T − benchmark kernel | " + c("d1t_cbig") + " |",
    ])


def filter_table(V, used):
    return "\n".join([
        "| Kernel | Without the filter | Filter from all subjects | Filter from training subjects only |",
        "|---|---|---|---|",
        "| Full correlations (no conditioning) | {{ra_fullw1_r:.4f}} | {{ra_full_filt_r:.4f}} | |",
        "| Conditioned (shrinkage, partial correlation, eigenvalue power, full-correlation mix) | {{ra_nofilt_r:.4f}} | **{{ra_canon_r:.4f}}** | {{ra_canon_ind_r:.4f}} |",
        "| The benchmark's kernel, double-centred and trace-normalised | {{ra_cbig6_hkh_r:.4f}} | {{ra_cbig6_filt_r:.4f}} | |",
    ])


def prereg_table(V, used):
    rules = {"seeds": "positive in at least {{pr_%s_need}} of {{pr_%s_of}} repeats",
             "p": "bootstrap P(difference > 0) at least 0.95",
             "ci": "95% interval above zero", "cells": "the addendum's rule in every cell",
             "t": "difference above zero and corrected t at least {{ext_tcrit:.3f}}"}
    out = ["| Pre-registered contrast | Rule as written | Observed | Met |", "|---|---|---|---|"]
    heads = {"single": "**Single-decision contrasts**", "multi": "**Contrasts with a rule per cell**",
             "secondary": "**Secondary contrasts declared with the main rule**", "status": "**Registered, not decided**"}
    meta = _meta()["prereg"]
    for grp in ("single", "multi", "secondary", "status"):
        out.append(f"| {heads[grp]} | | | |")
        for key, label, rule, g in meta:
            if g != grp:
                continue
            lab = label[0].upper() + label[1:]
            if grp == "status":
                obs = {"m": "motion index (DVARS) alone AUC {{pr_m_dvars_auc:.3f}}; margin {{pr_m_margin:.3f}} before and {{pr_m_margin_after:.3f}} after removing it; best closed-form temporal feature {{pr_m_best_temporal:.3f}} against F-BERT {{pr_m_fbert_auc:.3f}}",
                       "clip": "voided: every modality had been built from resting-state features",
                       "chainho": "superseded before it ran"}[key]
                verdict = {"m": "reading: temporal or motion signal the static connectome lacks", "clip": "—", "chainho": "—"}[key]
                out.append(f"| {lab} | {'reading rule fixed in advance' if key == 'm' else 'decision rule fixed in advance'} | {obs} | {verdict} |")
                continue
            r = rules[rule] % (key, key) if rule == "seeds" else rules[rule]
            if key == "d":
                r += ", and above both F-BERT probes"
            if rule == "seeds":
                o = f"{{{{pr_{key}_wins}}}} of {{{{pr_{key}_of}}}}"
            elif rule == "p":
                o = f"P = {{{{pr_{key}_p:.3f}}}}"
            elif rule == "ci":
                o = f"{{{{pr_{key}_lo:.3f}}}} to {{{{pr_{key}_hi:.3f}}}}"
            elif rule == "t":
                o = f"t = {{{{pr_{key}_t:.2f}}}}"
            elif key == "cbert":
                o = "behind F-BERT-8M by mean in {{pr_cbert_behind}} of {{pr_cbert_cells}} cells (per-repeat counts not stored)"
            else:
                o = f"{{{{pr_{key}_cells_met}}}} of {{{{pr_{key}_cells}}}} cells"
            out.append(f"| {lab} | {r} | {o} | {{{{y:pr_{key}_met}}}} |")
    return "\n".join(out)


def files_table(V, used):
    meta = _meta()
    scripts, read = meta["scripts"], meta["files_read"]
    files = {"results/" + f: h for f, h in read.items()}
    for r in _rows():                      # sources outside results/ (code constants, copied values)
        if not r["source_file"].startswith("results/") and r["source_file"] != "results/":
            files.setdefault(r["source_file"], r["sha16"])
    use, dup = meta.get("prior_use", {}), meta.get("prior_dup", {})
    out = ["| File | SHA-256 (16) | Produced by | Earlier use |", "|---|---|---|---|"]
    for f, h in sorted(files.items()):
        base = f.split("/")[-1]
        if base.startswith("FOLDNB_"):
            sc = "code/fbert/fold_nb.py, from the per-fold scores of the rerun"
        elif base.endswith("_FOLDS_1119.json"):
            sc = "code/fbert/fold_scores.py (rerun of the original script)"
        else:
            sc = scripts.get(base, "the code itself" if f.startswith("code/") else "not in the repository")
        if base == "FBERT_UNION_SUMMARY_1117.json":
            sc = "summary of the in-fold union arms of the cohort runs"
        if base in ("CANON_NUMBERS.json", "_EXPORTED_PRIVATE_VALUES.json"):
            sc = "values copied from the F-BERT preprint's table, or a file count"
        u = use.get(base, "")
        if base in dup:
            u += " (identical copy)"
        out.append(f"| {f} | {h} | {sc} | {u} |")
    return "\n".join(out)


def discussion_tally(V, used):
    return ("Of the {{pr_n}} pre-registered contrasts with a single decision (Table S2), {{pr_met}} were met and "
            "{{pr_notmet}} were not; {{pr_met_by_repeats}} of the {{pr_met}} met were decided by counting the repeats in which "
            "the difference was positive, which is not a test. The unmet ones are the six-modality comparison of the per-subject "
            "pipeline with F-BERT and two rounds that tried to rescue it; three refinements tested on held-out subjects (an average with "
            "a second extraction, a three-atlas average and a seventh task); dropping the motor task, within the benchmark splits "
            "and on held-out subjects; and ADHD-200 diagnosis. Of the contrasts with a rule per cell, the in-fold choice between D0 and "
            "F-BERT's target was met in {{pr_k_cells_met}} of {{pr_k_cells}} cells, the combined system in {{pr_n_cells_met}} of "
            "{{pr_n_cells}} cells and its last attempt on ADHD-200 in {{pr_o_cells_met}} of {{pr_o_cells}}; combined systems that "
            "included our own pre-trained encoders met their rule in none of their cells, and the encoder alone was behind F-BERT-8M by mean in "
            "{{pr_cbert_behind}} of {{pr_cbert_cells}} cells. Of {{pr_sec_n}} secondary contrasts "
            "declared with the main rule, {{pr_sec_met}} were met.")