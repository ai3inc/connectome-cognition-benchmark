"""Every verbal comparison in manuscript.src.md has a machine check here.
The description quotes or paraphrases the sentence it guards."""
import json, os

Z = 1.96  # two-sided 5% threshold for a corrected t with hundreds of degrees of freedom


def claims(V):
    g = V.__getitem__
    meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger", "tables_meta.json")))
    pr_keys = [k for k, _, _, grp in meta["prereg"] if grp == "single"]
    unmet = sorted(k for k in pr_keys if not g(f"pr_{k}_met"))
    pubs = [g(k) for k in ("pub_rs", "pub_lang", "pub_social", "pub_gamb", "pub_wm", "pub_motor", "pub_best",
                           "pub_multikrr", "pub_stack6", "pub_stackall")]
    C = [
        # headline
        ("stacking six is the best model in the benchmark paper", g("pub_stack6") == max(pubs)),
        ("ours fixed exceeds the best published model and the re-run kernel", g("canon_r") > g("pub_stack6") and g("canon_r") > g("cbig6_r")),
        ("the margin over the re-run kernel is significant (corrected t)", g("canon_vs_cbig6_t") > Z and g("canon_vs_cbig6_p") < 0.05),
        ("settings chosen in-fold also exceed the re-run kernel significantly", g("nested_vs_cbig6_t") > Z),
        ("re-choosing settings in-fold changes little", abs(g("nested_minus_fixed_t")) < 1),
        ("'ahead ... in all splits'", g("canon_vs_cbig6_seed_wins") == g("n_seeds")),
        ("naive split t far exceeds the corrected t", g("canon_vs_cbig6_seed_naive_t") > 10 * g("canon_vs_cbig6_t")),
        ("the in-fold rule's favourite cell is not the fixed one", g("posthoc_cell_r") != g("canon_r")),
        # scoring
        ("ridge-grid floor reverses the order of the unfiltered pipeline and the benchmark kernel (raw target)",
         g("nofilt_raw_oldgrid") < g("cbig6_raw_r") < g("nofilt_raw_r")),
        ("the unfiltered pipeline chose the smallest value in every fold", g("nofilt_lam_floor_folds") == g("n_fold_scores")),
        ("ridgeless limit: a higher floor changes the score by less than 0.001", abs(g("nofilt_raw_floor6") - g("nofilt_raw_r")) < 0.001),
        ("the filtered pipeline chose 1e-3 in every fold", g("canon_lam_interior_folds") == g("n_fold_scores")),
        ("re-run matches published single modalities closely on the raw target", g("rerun_single_maxdiff") < 0.01),
        ("under our covariate step the resting re-run is below the published value", g("rest_cbig_r") < g("pub_rs")),
        ("the six-modality re-run is above both published combinations", g("cbig6_r") > g("pub_stack6") and g("cbig6_r") > g("pub_multikrr")),
        # factorial and inductive filter
        ("filter lowers accuracy on full correlations (significant)", g("ra_filter_on_full_d") < 0 and g("ra_filter_on_full_t") < -Z),
        ("filter lowers accuracy on the benchmark kernel, not significantly",
         g("ra_filter_on_benchmark_kernel_d") < 0 and abs(g("ra_filter_on_benchmark_kernel_t")) < Z),
        ("filter helps only after conditioning (significant)", g("ra_filter_on_conditioned_d") > 0 and g("ra_filter_on_conditioned_t") > Z),
        ("the interaction is significant", g("ra_interaction_t") > Z),
        ("conditioning without the filter is not significant", abs(g("ra_conditioning_without_filter_t")) < Z),
        ("inductive filter: no gain over no filter", g("ra_inductive_filter_vs_none_d") < 0 or abs(g("ra_inductive_filter_vs_none_t")) < Z),
        ("inductive filter: no significant gain over the benchmark kernel", abs(g("ra_canon_ind_vs_cbig6_t")) < Z),
        ("the transductive filter beats the inductive one significantly", g("ra_transductive_vs_inductive_filter_t") > Z),
        ("our full-correlation kernel equals the benchmark kernel (|t| < 1)", abs(g("ra_fullw1_vs_cbig6_t")) < 1),
        ("shrinkage and partial correlation: positive, not significant", g("ra_partial_without_power_d") > 0 and g("ra_partial_without_power_t") < Z),
        ("the power step only hurt (significant loss)", g("ra_power_step_d") < 0 and g("ra_power_step_t") < -Z),
        ("the mix step is positive", g("ra_mix_step_d") > 0),
        ("language weighting adds nothing measurable", abs(g("ra_language_weight_t")) < 1),
        ("with equal weights the margin stays significant", g("ra_canon_eq_vs_cbig6_t") > Z),
        ("review arms reproduce the headline run", g("ra_canon_r") == g("canon_r") and g("ra_nofilt_r") == g("nofilt_r") and g("ra_cbig6_r") == g("cbig6_r")),
        ("second implementation reproduces the headline", g("clip_cert_r") == g("canon_r")),
        ("without the filter the six-modality margin is not significant", g("nofilt_vs_cbig6_t") < Z),
        # resting state
        ("resting state: conditioning without the filter is significant, unlike the six-modality ensemble",
         g("rest_nofilt_vs_cbig_t") > Z and g("nofilt_vs_cbig6_t") < Z),
        ("resting state: ours without filter above published and re-run", g("rest_nofilt_r") > g("pub_rs") > g("rest_cbig_r")),
        # new subjects
        ("sanity gate passed and the verdict is CONFIRMED", g("ns_gate") and g("ns_confirmed")),
        ("primary interval above zero", g("ns_prim_lo") > 0),
        ("unfiltered variant also above the benchmark kernel", g("ns_d0_vs_cbig6_lo") > 0),
        ("margins against the plain kernel are larger", g("ns_d1_vs_plain_t_d") > g("ns_prim_d") and g("ns_d0_vs_plain_i_d") > g("ns_d0_vs_cbig6_d")),
        ("filter share on new subjects not resolved", g("ns_d1_vs_d0_lo") < 0 < g("ns_d1_vs_d0_hi")),
        ("margin held among the least similar subjects", g("ns_clean_prim_lo") > 0),
        ("figure 3: all intervals above zero except D1 minus D0",
         min(g("ns_prim_lo"), g("ns_d0_vs_cbig6_lo"), g("ns_d1_vs_plain_t_lo"), g("ns_d0_vs_plain_i_lo"),
             g("ns_d1_vs_fbert_lo"), g("ns_d0_vs_fbert_lo")) > 0 and g("ns_d1_vs_d0_lo") < 0),
        ("both new-subject arms picked an edge of the shrinkage grid", g("ns_d0_g") == g("ns_g_max") and g("ns_d1_g") == g("ns_g_min")),
        ("every arm scored higher on the new subjects than in-sample on the same extraction or features",
         g("ns_d1_r") > g("fx_d1") and g("ns_d0_r") > g("fx_d0") and g("ns_fbert_r") > g("fx_fbert_cat") and g("ns_cbig6_r") > g("cbig6_raw_r")
         and g("ns_plain_t_r") > g("fx_plain") and g("ns_plain_i_r") > g("fx_plain")),
        ("benchmark kernel: same score at three ridge floors", g("cbig6_raw_oldgrid") == g("cbig6_raw_floor6") == g("cbig6_raw_r")),
        ("on our extraction D0 chose the top shrinkage in most folds, which equals the new-subject choice",
         g("six_d0_g_top_folds") > 0.9 * g("six_d0_g_folds") and g("ns_d0_g") == 0.98),
        ("pooled in-sample conditioning margin also close to the new-subject margin", abs(g("six_d0_plain_pooled_d") - g("ns_d0_vs_plain_i_d")) < 0.01),
        ("raw-target conditioning gain on benchmark features also not significant", abs(g("ra_raw_conditioning_without_filter_t")) < Z),
        ("other components: below the plain kernel, not significant", g("other_er_t") < 0 and abs(g("other_er_t")) < Z and abs(g("other_sat_t")) < Z),
        ("external cohorts: the filter scored lower in all three, significantly only in AOMIC (ABIDE not significant; ADHD-200 untested)",
         g("fe_aomic_d1_d0_t") < -Z and abs(g("fe_abide_d1_d0_t")) < Z and abs(g("fe_abide_fiq_d1_d0_t")) < Z),
        ("most of the new subjects were in the early cross-validated cohort", g("pool_new_in_pool") > g("ns_hold") / 2),
        ("levers: penalty averaging lowered accuracy clearly; the kernel replacement only nominally",
         g("lev_lambag_t") < -Z and -2.5 < g("lev_alpha_t") < -Z),
        ("the C-BERT-alone every-cell rule fails: behind F-BERT by mean in at least one cell", g("pr_cbert_behind") > 0 and not g("pr_cbert_met")),
        ("two hand-set mixtures nominally significant; the eigenvalue-power kernel mixed in at small weight changed nothing",
         g("lev_rankmix_t") > Z and g("lev_clipmix_t") > Z and abs(g("lev_alphamix_t")) < 1),
        ("D0 and the benchmark kernel chose the ridge floor", g("ns_d0_lam") == g("ns_lam_floor") and g("ns_cbig_lam") == g("ns_lam_floor")),
        ("the earlier 233-subject run improved after the grid was widened", g("c233_before_grid_fix") < g("c233_d1")),
        ("the filter lowered accuracy in all three external cohorts",
         g("aomic_d1") < g("aomic_d0") and g("abide_d1") < g("abide_d0") and g("abide_fiq_d1") < g("abide_fiq_d0")
         and g("adhd_d1") < g("adhd_d0") and g("adhd_iq_d1") < g("adhd_iq_d0")),
        # foundation model (exact fold-level tests, Table 4)
        ("rest: D0 beats F-BERT-8M (corrected t)", g("fe_rest_d0_fbert_t") > Z and g("fe_rest_d0_fbert_d") > 0),
        ("rest: D0 tied F-BERT's pre-training target", abs(g("fe_rest_d0_teacher_t")) < 1),
        ("rest: the target beat the encoder by the same margin",
         g("fe_rest_teacher_fbert_t") > Z and abs(g("fe_rest_teacher_fbert_d") - g("fe_rest_d0_fbert_d")) < 0.001),
        ("rest: the encoder was ahead of a plain correlation kernel", g("fe_rest_plain_fbert_t") < -Z),
        ("rest: the filter did not help", g("fe_rest_d1_d0_d") < 0 and abs(g("fe_rest_d1_d0_t")) < Z),
        ("six: D0 and D1 led F-BERT, neither significantly",
         0 < g("fe_six_d0_fbert_t") < Z and 0 < g("fe_six_d1_fbert_t") < Z and g("fe_six_d0_fbert_d") > 0 and g("fe_six_d1_fbert_d") > 0),
        ("six, pooled per split: D0 behind F-BERT and contrast A not met", g("six_d0_fbert_d") < 0 and not g("pr_a_met")),
        ("six: the scoring choice changed the ordering", g("six_d0_fbert_d") < 0 < g("fe_six_d0_fbert_d")),
        ("new subjects: both variants beat F-BERT", g("ns_d0_vs_fbert_lo") > 0 and g("ns_d1_vs_fbert_lo") > 0),
        ("ABIDE diagnosis: D0 beat F-BERT", g("fe_abide_d0_fbert_t") > Z),
        ("AOMIC: D0 beat F-BERT, just above the threshold", Z < g("fe_aomic_d0_fbert_t") < 2.5),
        ("abstract: AOMIC nominal, p between 0.01 and 0.05", 0.01 < g("fe_aomic_d0_fbert_p") < 0.05),
        ("abstract: ABIDE IQ not significant", abs(g("fe_abide_fiq_d0_fbert_t")) < Z),
        ("short scan: all splits ahead (rule met) but corrected t not significant; level with F-BERT-1M; F-BERT-8M beats plain FC",
         g("pr_sec4_wins") == g("pr_sec4_of") and g("pr_sec4_met") and 0 < g("fe_short_d0_fbert_t") < Z and abs(g("fe_short_d0_fbert1_t")) < 1
         and g("fe_short_plain_fbert_t") < -Z),
        ("NYU diagnostic: motion index alone low, less than half the margin removed, no temporal feature reaches F-BERT",
         g("pr_m_dvars_auc") < 0.6 and g("pr_m_margin_after") > g("pr_m_margin") / 2 and g("pr_m_best_temporal") < g("pr_m_fbert_auc")),
        ("own encoder ahead by mean in some cells", 0 < g("pr_cbert_ahead") < g("pr_cbert_cells")),
        ("AOMIC: the filter lowered accuracy significantly", g("fe_aomic_d1_d0_d") < 0 and g("fe_aomic_d1_d0_t") < -Z),
        ("abstract: not significant with six modalities or on ADHD-200",
         abs(g("fe_six_d0_fbert_t")) < Z and g("adhd_lo") < 0 < g("adhd_hi") and g("adhd_iq_lo") < 0 < g("adhd_iq_hi")),
        ("ABIDE IQ: ahead in all repeats but not significantly",
         g("fe_abide_fiq_d0_fbert_pos") == g("fe_abide_fiq_splits") and 0 < g("fe_abide_fiq_d0_fbert_t") < Z),
        ("ADHD-200: neither difference resolved", g("adhd_lo") < 0 < g("adhd_hi") and g("adhd_iq_lo") < 0 < g("adhd_iq_hi")),
        ("NYU post hoc: F-BERT ahead on diagnosis", g("nyu_dx_fbert") > g("nyu_dx_d0")),
        ("every external cohort: filtered below unfiltered (per-fold means and pooled scores)",
         g("fe_aomic_m_d1") < g("fe_aomic_m_d0") and g("fe_abide_m_d1") < g("fe_abide_m_d0") and g("fe_abide_fiq_m_d1") < g("fe_abide_fiq_m_d0")
         and g("aomic_d1") < g("aomic_d0") and g("abide_d1") < g("abide_d0") and g("abide_fiq_d1") < g("abide_fiq_d0")
         and g("adhd_d1") < g("adhd_d0") and g("adhd_iq_d1") < g("adhd_iq_d0")),
        # reconciliation of the conditioning gain
        ("conditioning gained more on new subjects than on the benchmark's features",
         g("ns_d0_vs_cbig6_d") > g("ra_conditioning_without_filter_d") and g("ns_d0_vs_plain_i_d") > g("ra_conditioning_without_filter_d")),
        ("on our extraction in-sample, D0 led the plain kernel significantly, close to its new-subject margin",
         g("fe_six_d0_plain_t") > Z and abs(g("fe_six_d0_plain_d") - g("ns_d0_vs_plain_i_d")) < 0.01),
        # post hoc levers and other targets
        ("exactly two post hoc changes improved nominally (the mixtures) and two lowered it significantly", g("lev_pos_sig") == 2 and g("lev_neg_sig") == 2),
        ("the two significant losses are the penalty averaging and the kernel replacement",
         g("lev_lambag_t") < -Z and g("lev_alpha_t") < -Z),
        ("clipping is the largest positive change", g("lev_clip_d") == max(g(f"lev_{k}_d") for k, *_ in meta["levers"])),
        ("RELATIONAL not significant", g("lev_rel_t") < Z),
        ("on the other two components the chain is below the plain kernel",
         g("other_er_chain") < g("other_er_plain") and g("other_sat_chain") < g("other_sat_plain")),
        # registry
        ("tally adds up", g("pr_met") + g("pr_notmet") == g("pr_n")),
        ("discussion lists exactly the unmet single-decision contrasts", unmet == sorted(["a", "b", "c", "f", "g", "l", "r", "s", "t2"])),
        ("combined systems with our own encoders met none of their cells (build_numbers.py asserts that no cell could pass)",
         all(g(f"pr_{k}_cells_met") == 0 for k in ("p", "q1", "q2", "q3"))),
        ("Addendum M reading: neither the motion nor the closed-form temporal branch was triggered",
         not g("pr_m_dvars_alone_ok") and not g("pr_m_half_removed") and not g("pr_m_temporal_reaches")),
        ("N met in some but not all cells", 0 < g("pr_n_cells_met") < g("pr_n_cells")),
    ]
    return C
