"""Build ledger/numbers_index.json and the figures from ../results/*.json.

Every number in manuscript.src.md comes from here. Nothing is typed by hand: each row
records the source file, its sha256 prefix and a locator. Nadeau-Bengio t values for
the benchmark comparisons are recomputed from the stored per-fold scores and checked
against the values the result files recorded (assertions), so a drift fails the build.

usage: python3 -B -I build_numbers.py        (run from anywhere)
"""
import hashlib, json, math, os, re, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ROWS, SEEN = [], set()
_cache, _sha = {}, {}


def load(f):
    if f not in _cache:
        p = os.path.join(RES, f)
        b = open(p, "rb").read()
        _sha[f] = hashlib.sha256(b).hexdigest()[:16]
        _cache[f] = json.loads(b)
    return _cache[f]


def dig(f, path):
    d = load(f)
    for k in path:
        d = d[k]
    return d


def put(key, value, f, locator):
    assert key not in SEEN, f"duplicate key {key}"
    SEEN.add(key)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    load(f)
    ROWS.append({"key": key, "value": value, "source_file": "results/" + f, "sha16": _sha[f], "locator": locator})
    return value


def take(key, f, *path, scale=1.0, nd=None):
    v = dig(f, path)
    if scale != 1.0:
        v = v * scale
    if nd is not None:
        v = round(float(v), nd)
    return put(key, v, f, "$." + ".".join(str(p) for p in path))


def nb(pa, pb, ratio):
    """Nadeau-Bengio corrected resampled t over per-fold differences (all folds of all seeds)."""
    d = (np.asarray(pa) - np.asarray(pb)).ravel()
    m = len(d)
    tc = d.mean() / math.sqrt((1.0 / m + ratio) * d.var(ddof=1))
    return float(d.mean()), float(tc), float(2 * stats.t.sf(abs(tc), m - 1)), m


def seed_factor(J, k=10):
    """Per-seed approximation used by code/fbert/corrected_t_review.py (fold scores not saved)."""
    return math.sqrt((1.0 / (k * J) + 1.0 / (k - 1)) * k * J)


# ---------------------------------------------------------------- benchmark (Ooi features)
MC = "METRIC_CHECK_COV753_1118.json"
pf = load(MC)["per_fold_r"]
ratio = take("ratio_test_train", MC, "test_train_ratio")
take("n_bench", MC, "n")
take("n_cov", MC, "n_subset")
arr = np.array(pf["B_sub|resid|CANON_lang2"])
put("n_seeds", arr.shape[0], MC, "$.per_fold_r['B_sub|resid|CANON_lang2'] rows")
put("n_folds", arr.shape[1], MC, "$.per_fold_r['B_sub|resid|CANON_lang2'] columns")
put("n_fold_scores", arr.size, MC, "rows x columns")
put("nb_factor", round(math.sqrt(1 + arr.size * ratio), 2), MC, "sqrt(1 + folds x ratio)")

ARMS = {"canon": "CANON_lang2", "nofilt": "NOFILT_lang2", "tauonly": "TAUONLY_lang2",
        "noalpha": "NOALPHA_NOFILT_lang2", "cbigpart": "CBIGPART_lang2", "cbigpart_noalpha": "CBIGPART_NOALPHA_lang2",
        "cbig6": "CBIG6_eq"}
for k, a in ARMS.items():
    take(f"{k}_r", MC, "summary", f"B_sub|resid|{a}", "mean_perfold_r")
    take(f"{k}_sd", MC, "summary", f"B_sub|resid|{a}", "perfold_seed_sd")
take("canon_pooled_resid", MC, "summary", "B_sub|resid|CANON_lang2", "pooled_r")
take("cbig6_pooled_resid", MC, "summary", "B_sub|resid|CBIG6_eq", "pooled_r")
take("canon_raw_r", MC, "summary", "A753|raw|CANON_lang2", "mean_perfold_r")
take("canon_raw_pooled", MC, "summary", "A753|raw|CANON_lang2", "pooled_r")
take("cbig6_raw_r", MC, "summary", "A753|raw|CBIG6_eq", "mean_perfold_r")
take("cbig6_raw_pooled", MC, "summary", "A753|raw|CBIG6_eq", "pooled_r")
take("nofilt_raw_r", MC, "summary", "A753|raw|NOFILT_lang2", "mean_perfold_r")
take("canon_fold_sd", MC, "summary", "B_sub|resid|CANON_lang2", "perfold_fold_sd")

PAIRS = {"canon_vs_cbig6": ("CANON_lang2", "CBIG6_eq"), "nofilt_vs_cbig6": ("NOFILT_lang2", "CBIG6_eq"),
         "filter": ("CANON_lang2", "NOFILT_lang2"), "tauonly_vs_cbig6": ("TAUONLY_lang2", "CBIG6_eq"),
         "keep_trunc": ("CANON_lang2", "TAUONLY_lang2"), "alpha_nofilt": ("NOFILT_lang2", "NOALPHA_NOFILT_lang2"),
         "kernelform": ("NOFILT_lang2", "CBIGPART_lang2")}
for k, (a, b) in PAIRS.items():
    d, t, p, m = nb(pf[f"B_sub|resid|{a}"], pf[f"B_sub|resid|{b}"], ratio)
    put(f"{k}_diff", round(d, 4), MC, f"per_fold_r resid {a} - {b}")
    put(f"{k}_t", round(t, 2), MC, f"NB t, per_fold_r resid {a} vs {b}")
    put(f"{k}_p", float(f"{p:.3g}"), MC, f"NB p, per_fold_r resid {a} vs {b}")
    stored = load(MC)["nadeau_bengio_exact_subset_resid"].get(f"{a}_vs_CBIG6")
    if b == "CBIG6_eq" and stored:
        assert abs(stored["corrected_t"] - round(t, 2)) < 0.011, (k, stored["corrected_t"], t)
# wins counted per fold
d = np.array(pf["B_sub|resid|CANON_lang2"]) - np.array(pf["B_sub|resid|CBIG6_eq"])
put("canon_vs_cbig6_fold_wins", int((d > 0).sum()), MC, "folds with canon > cbig6")
dseed = d.mean(1)
put("canon_vs_cbig6_seed_wins", int((dseed > 0).sum()), MC, "seeds with canon > cbig6")
naive_seed_t = dseed.mean() / (dseed.std(ddof=1) / math.sqrt(len(dseed)))
put("canon_vs_cbig6_seed_naive_t", round(float(naive_seed_t), 1), MC, "naive t over seed means")

# the benchmark's own published numbers (their results files, read by metric_check.py)
OOI = ["ooi_own_results", "KRR_single:results.fmri"]
for m in ["rs", "lang", "social", "gamb", "wm", "motor"]:
    take(f"pub_{m}", MC, *OOI, m, "mean")
CMB = ["ooi_own_results", "combined:results.combined"]
take("pub_best", MC, *CMB, "best", "mean")
take("pub_multikrr", MC, *CMB, "multiKRR_six", "mean")
take("pub_stack6", MC, *CMB, "stacking_LRR_six", "mean")
take("pub_stack6_sd", MC, *CMB, "stacking_LRR_six", "sd")
take("pub_stackall", MC, *CMB, "stacking_LRR_all", "mean")
V = {r["key"]: r["value"] for r in ROWS}
put("canon_vs_pub", round(V["canon_r"] - V["pub_stack6"], 4), MC, "canon_r - pub_stack6")
put("nofilt_vs_pub", round(V["nofilt_r"] - V["pub_stack6"], 4), MC, "nofilt_r - pub_stack6")
put("cbig6_vs_pub", round(V["cbig6_r"] - V["pub_stack6"], 4), MC, "cbig6_r - pub_stack6")
put("canon_vs_pub_multikrr", round(V["canon_r"] - V["pub_multikrr"], 4), MC, "canon_r - pub_multikrr")
put("cov_cost_canon", round(V["canon_raw_r"] - V["canon_r"], 4), MC, "raw - resid, canon")
put("pooled_minus_perfold_canon_raw", round(V["canon_raw_pooled"] - V["canon_raw_r"], 4), MC, "pooled - per-fold, raw")

# re-run of their kernel, single modalities, raw target, canonical grid file
MO = "METRIC_CHECK_1118.json"
diffs = []
for m in ["rs", "lang", "social", "gamb", "motor", "wm"]:
    v = take(f"rerun_{m}", MO, "summary", f"A753|raw|CBIG_{m}", "mean_perfold_r")
    diffs.append(abs(v - V[f"pub_{m}"]))
put("rerun_single_maxdiff", round(max(diffs), 4), MO, "max |rerun_m - pub_m| over six fMRI modalities")
# lambda floor
take("nofilt_raw_oldgrid", MO, "summary", "A753|raw|NOFILT_lang2", "mean_perfold_r")
take("canon_raw_oldgrid", MO, "summary", "A753|raw|CANON_lang2", "mean_perfold_r")
import re as _re
_cp = os.path.join(HERE, "..", "code", "canonical.py")
_src = open(_cp, "rb").read()
_lams = eval(_re.search(rb"^LAMBDAS = (\(.*\))", _src, _re.M).group(1).decode())
SEEN.add("lam_floor_old")
ROWS.append({"key": "lam_floor_old", "value": float(min(_lams)), "source_file": "code/canonical.py",
             "sha16": hashlib.sha256(_src).hexdigest()[:16], "locator": "LAMBDAS min"})
put("lam_floor_new", float(min(load(MC)["lambda_grid"])), MC, "$.lambda_grid min")
V = {r["key"]: r["value"] for r in ROWS}
put("lam_floor_understatement", round(V["nofilt_raw_r"] - V["nofilt_raw_oldgrid"], 4), MO, "nofilt raw new grid - old grid")
take("nofilt_raw_floor6", "METRIC_CHECK_LAMGRID_1118.json", "summary", "A753|raw|NOFILT_lang2", "mean_perfold_r")
take("lam_floor6", "METRIC_CHECK_LAMGRID_1118.json", "lambda_grid", 0)
_exp = os.path.join(RES, "_EXPORTED_PRIVATE_VALUES.json")
if os.path.exists(_exp):      # public bundle: values that need the private repository, exported at release
    _ev_ = json.load(open(_exp))
    _nres, _src_n = _ev_["n_result_files"], "results/_EXPORTED_PRIVATE_VALUES.json"
else:
    _nres, _src_n = len([f for f in os.listdir(RES) if f.endswith(".json")]), "results/"
ROWS.append({"key": "n_result_files", "value": _nres, "source_file": _src_n, "sha16": "-", "locator": "count of results/*.json in the private repository"}); SEEN.add("n_result_files")
lc = load(MC)["lambda_chosen"]
put("canon_lam_interior_folds", lc["B_sub|resid|CANON_lang2"].get("0.001", 0), MC, "$.lambda_chosen resid CANON at 1e-3")
put("nofilt_lam_floor_folds", lc["B_sub|resid|NOFILT_lang2"].get("1e-12", 0), MC, "$.lambda_chosen resid NOFILT at floor")
put("cbig6_lam_floor_folds", lc["B_sub|resid|CBIG6_eq"].get("1e-12", 0), MC, "$.lambda_chosen resid CBIG6 at floor")
take("cbig6_raw_oldgrid", MO, "summary", "A753|raw|CBIG6_eq", "mean_perfold_r")
take("cbig6_raw_floor6", "METRIC_CHECK_LAMGRID_1118.json", "summary", "A753|raw|CBIG6_eq", "mean_perfold_r")


# ---------------------------------------------------------------- method constants (code/canonical.py)
def put_code(key, value, locator, rel="code/canonical.py"):
    assert key not in SEEN, key
    SEEN.add(key)
    b = open(os.path.join(HERE, "..", rel), "rb").read()
    ROWS.append({"key": key, "value": value, "source_file": rel,
                 "sha16": hashlib.sha256(b).hexdigest()[:16], "locator": locator})


for name, key in [("G_SHRINK", "c_g"), ("W_FULL", "c_wfull"), ("TAU", "c_tau"), ("KEEP", "c_keep"), ("ALPHA", "c_alpha")]:
    m = _re.search(rb"^" + name.encode() + rb" = ([0-9.]+)", _src, _re.M)
    put_code(key, float(m.group(1)) if b"." in m.group(1) else int(m.group(1)), name)
put_code("c_wfull_pct", int(round(100 * float(_re.search(rb"^W_FULL = ([0-9.]+)", _src, _re.M).group(1)))), "100 x W_FULL")
put_code("c_lang_w", int(_re.search(rb"MOD_WEIGHTS = \{m: \((\d+) if m == 'lang'", _src).group(1)), "MOD_WEIGHTS language")
put_code("c_inner", int(_re.search(rb"np.array_split\(idx, (\d+)\)", _src).group(1)), "_inner_splits")
_eh = open(os.path.join(HERE, "..", "code", "fbert", "eval_holdout.py"), "rb").read()
put_code("c_holdout_inner", int(_re.search(rb"^INNER_K = (\d+)", _eh, _re.M).group(1)), "INNER_K", "code/fbert/eval_holdout.py")
_ec = open(os.path.join(HERE, "..", "code", "fbert", "eval_holdout_chain.py"), "rb").read()
put_code("c_leak_pct", int(_re.search(rb"np\.percentile\(Ctt\.max\(1\), (\d+)\)", _ec).group(1)), "leak-control percentile", "code/fbert/eval_holdout_chain.py")
put_code("c_boot", int(_re.search(rb"def boot_diff\(pa, pb, yt, n=(\d+)", _eh).group(1)), "boot_diff n", "code/fbert/eval_holdout.py")
_es = open(os.path.join(HERE, "..", "code", "fbert", "eval_samefolds.py"), "rb").read()
put_code("c_nreg_ours", int(_re.search(rb"^N_REG = (\d+)", _es, _re.M).group(1)), "N_REG", "code/fbert/eval_samefolds.py")
put_code("c_fbert_pow", float(_re.search(rb"np\.clip\(w, 0, None\) \*\* ([0-9.]+)\)", _es).group(1)), "p035 eigenvalue power (F-BERT's target)", "code/fbert/eval_samefolds.py")
put_code("c_short_tp", int(_re.search(rb"^SHORT_T = (\d+)", open(os.path.join(HERE, "..", "code", "fbert", "hcp_prep.py"), "rb").read(), _re.M).group(1)), "SHORT_T", "code/fbert/hcp_prep.py")
put_code("c_nreg", int(_re.search(rb"^IU = np.triu_indices\((\d+), 1\)", _src, _re.M).group(1)), "IU regions")

# nested selection under the same metric and covariates
NM = "NESTED_MATCHED_1118.json"
take("nested_r", NM, "mean_perfold_r", "NESTED")
take("nested_sd", NM, "seed_sd", "NESTED")
take("nested_minus_fixed", NM, "NB", "NESTED_vs_fixed_g0.92_k150", "diff_mean_perfold_r")
take("nested_minus_fixed_t", NM, "NB", "NESTED_vs_fixed_g0.92_k150", "corrected_t")
take("nested_minus_fixed_p", NM, "NB", "NESTED_vs_fixed_g0.92_k150", "p_two_sided")
take("nested_vs_cbig6", NM, "NB", "NESTED_vs_CBIG6", "diff_mean_perfold_r")
take("nested_vs_cbig6_t", NM, "NB", "NESTED_vs_CBIG6", "corrected_t")
take("nested_vs_cbig6_p", NM, "NB", "NESTED_vs_CBIG6", "p_two_sided")
take("nested_pick_top", NM, "picks", "CANON_g0.9_k150")
take("posthoc_cell_r", NM, "mean_perfold_r", "CANON_g0.9_k150")
put("nested_grid_cells", len(load(NM)["grid"]["g"]) * len(load(NM)["grid"]["keep"]), NM, "|g| x |keep|")
put("nested_grid_lams", len(load(NM)["grid"]["lambda"]), NM, "|lambda|")
put("nested_g_lo", min(load(NM)["grid"]["g"]), NM, "$.grid.g min")
put("nested_g_hi", max(load(NM)["grid"]["g"]), NM, "$.grid.g max")
put("nested_k_lo", min(load(NM)["grid"]["keep"]), NM, "$.grid.keep min")
put("nested_k_hi", max(load(NM)["grid"]["keep"]), NM, "$.grid.keep max")
put("lam_grid_max", float(max(load(MC)["lambda_grid"])), MC, "$.lambda_grid max")
put("lam_grid_n", len(load(MC)["lambda_grid"]), MC, "len($.lambda_grid)")
put("nested_vs_pub", round(dig(NM, ["mean_perfold_r", "NESTED"]) - V["pub_stack6"], 4), NM, "nested - pub_stack6")
assert abs(dig(NM, ["mean_perfold_r", "CANON_g0.92_k150"]) - V["canon_r"]) < 1e-9

# resting state alone
MR = "METRIC_CHECK_COV753_REST_1118.json"
for k, a in {"rest_canon": "CANON_rs", "rest_nofilt": "NOFILT_rs", "rest_cbig": "CBIG_rs"}.items():
    take(f"{k}_r", MR, "summary", f"B_sub|resid|{a}", "mean_perfold_r")
take("rest_cbig_raw_r", MR, "summary", "A753|raw|CBIG_rs", "mean_perfold_r")
pr = load(MR)["per_fold_r"]
for k, (a, b) in {"rest_canon_vs_cbig": ("CANON_rs", "CBIG_rs"), "rest_nofilt_vs_cbig": ("NOFILT_rs", "CBIG_rs")}.items():
    d, t, p, m = nb(pr[f"B_sub|resid|{a}"], pr[f"B_sub|resid|{b}"], load(MR)["test_train_ratio"])
    put(f"{k}_diff", round(d, 4), MR, f"per_fold_r resid {a} - {b}")
    put(f"{k}_t", round(t, 2), MR, f"NB t {a} vs {b}")
    put(f"{k}_p", float(f"{p:.2g}"), MR, f"NB p {a} vs {b}")
V = {r["key"]: r["value"] for r in ROWS}
put("rest_nofilt_vs_pub", round(V["rest_nofilt_r"] - V["pub_rs"], 4), MR, "rest_nofilt_r - pub_rs")
put("rest_canon_vs_pub", round(V["rest_canon_r"] - V["pub_rs"], 4), MR, "rest_canon_r - pub_rs")

# ---------------------------------------------------------------- held-out subjects (pre-registered)
NS = "CHAIN_VS_OOI_KERNEL_263_1119.json"
take("ns_train", NS, "n_train")
take("ns_hold", NS, "n_holdout")
take("ns_hold_clean", NS, "n_holdout_leak_controlled")
put("ns_gate", bool(load(NS)["gate_passed"]), NS, "$.gate_passed")
put("ns_confirmed", load(NS)["verdict"] == "CONFIRMED", NS, "$.verdict")
for a in ["D1", "D0", "CBIG6", "PLAIN_T", "PLAIN_I"]:
    take(f"ns_{a.lower()}_r", NS, "holdout_r", a, nd=4)
take("ns_prim_d", NS, "primary_D1_vs_CBIG6", "diff", nd=3)
take("ns_prim_lo", NS, "primary_D1_vs_CBIG6", "ci95", 0, nd=3)
take("ns_prim_hi", NS, "primary_D1_vs_CBIG6", "ci95", 1, nd=3)
for k in ["D0_vs_CBIG6", "D1_vs_PLAIN_T", "D0_vs_PLAIN_I", "D1_vs_D0"]:
    kk = "ns_" + k.lower()
    take(kk + "_d", NS, "secondary", k, "diff", nd=3)
    take(kk + "_lo", NS, "secondary", k, "ci95", 0, nd=3)
    take(kk + "_hi", NS, "secondary", k, "ci95", 1, nd=3)
take("ns_clean_prim_d", NS, "secondary_leak_controlled", "D1_vs_CBIG6", "diff", nd=3)
take("ns_clean_prim_lo", NS, "secondary_leak_controlled", "D1_vs_CBIG6", "ci95", 0, nd=3)
take("ns_clean_prim_hi", NS, "secondary_leak_controlled", "D1_vs_CBIG6", "ci95", 1, nd=3)
FH = "FBERT_HOLDOUT_1117.json"
take("ns_fbert_r", FH, "holdout_r", "FBERT_best", nd=4)
take("ns_d1_vs_fbert_d", FH, "paired_vs_FBERT_best", "D1", "diff", nd=3)

# ---------------------------------------------------------------- independent cohort (AABC Release 2, pre-registered)
EX, EXS, EXN = "EXT_AABC_1119.json", "EXT_AABC_SAMPLE_1119.json", "EXT_AABC_NULLCHECK_1119.json"
_ex = load(EX)
assert _ex["verdict"] == "CONFIRMED" and _ex["info"]["mode"] == "unblinded"
assert load(EXS)["n"] == _ex["info"]["n"] and load(EXS)["sites"] == _ex["info"]["sites"]      # the sample file applies the same rule
assert load(EXS)["fc_sha16"] == _ex["info"]["fc_sha16"] and load(EXS)["pheno_sha16"] == _ex["info"]["pheno_sha16"]
take("ext_v1", EX, "info", "visits_V1")
take("ext_n", EX, "info", "n")
take("ext_excl", EX, "info", "excluded", "missing_target_or_covariate")
take("ext_fam", EX, "info", "families")
put("ext_sites", len(_ex["info"]["sites"]), EX, "len($.info.sites)")
take("ext_parcels", EXS, "parcels")
take("ext_edges", EXS, "edges")
take("ext_age_min", EXS, "age_min", nd=0)
take("ext_age_max", EXS, "age_max", nd=0)
take("ext_age_mean", EXS, "age_mean", nd=1)
take("ext_age_sd", EXS, "age_sd", nd=1)
take("ext_female_pct", EXS, "female_share", scale=100.0, nd=1)
_st = _ex["settings"]
take("ext_reps", EX, "settings", "N_REP")
take("ext_folds", EX, "settings", "N_FOLD")
take("ext_inner", EX, "settings", "INNER")
put("ext_g_n", len(_st["G_GRID"]), EX, "len($.settings.G_GRID)")
take("ext_g_lo", EX, "settings", "G_GRID", 0)
take("ext_g_hi", EX, "settings", "G_GRID", len(_st["G_GRID"]) - 1)
put("ext_lam_n", len(_st["LAMBDAS"]), EX, "len($.settings.LAMBDAS)")
take("ext_lam_lo", EX, "settings", "LAMBDAS", 0)
take("ext_lam_hi", EX, "settings", "LAMBDAS", len(_st["LAMBDAS"]) - 1)
take("ext_ratio", EX, "test_train_ratio", nd=4)
put("ext_n_fold_scores", _st["N_REP"] * _st["N_FOLD"], EX, "N_REP x N_FOLD")
put("ext_tcrit", round(float(stats.t.ppf(0.975, _st["N_REP"] * _st["N_FOLD"] - 1)), 3), EX, "two-sided 5% critical t, df = folds - 1 (pre-registered 1.972)")
assert abs(stats.t.ppf(0.975, _st["N_REP"] * _st["N_FOLD"] - 1) - 1.972) < 5e-4    # the value written in the pre-registration
for a, k in [("CBIG", "cbig"), ("PLAIN_I", "plain"), ("D0_I", "d0"), ("D1_I", "d1i"), ("D1_T", "d1t")]:
    take(f"ext_r_{k}", EX, "mean_perfold_r", a, nd=4)
    assert abs(np.mean(_ex["per_fold"][a]) - _ex["mean_perfold_r"][a]) < 6e-5, a
_ef = {a: np.asarray(_ex["per_fold"][a], dtype=float) for a in _ex["per_fold"]}
for key, (a, b), src in [("ext_d0_cbig", ("D0_I", "CBIG"), ["primary", "D0_I_vs_CBIG"]),
                         ("ext_d0_plain", ("D0_I", "PLAIN_I"), ["secondary", "D0_I_vs_PLAIN_I"]),
                         ("ext_d1i_d0", ("D1_I", "D0_I"), ["secondary", "D1_I_vs_D0_I"]),
                         ("ext_d1t_d0", ("D1_T", "D0_I"), ["secondary", "D1_T_vs_D0_I"]),
                         ("ext_d1t_cbig", ("D1_T", "CBIG"), ["secondary", "D1_T_vs_CBIG"])]:
    d_, t_, p_, m_ = nb(_ef[a], _ef[b], _ex["test_train_ratio"])
    rec = dig(EX, src)
    assert m_ == rec["folds"] and abs(t_ - rec["corrected_t"]) < 2e-3 and abs(d_ - rec["diff"]) < 6e-5, (key, t_, rec)
    put(key + "_d", round(d_, 4), EX, "$." + ".".join(src) + ".diff (recomputed from per_fold)")
    put(key + "_t", round(t_, 2), EX, "$." + ".".join(src) + ".corrected_t (recomputed from per_fold)")
    put(key + "_p", p_, EX, "two-sided p of the corrected t, df = folds - 1")
    put(key + "_pos", int(((_ef[a] - _ef[b]).mean(1) > 0).sum()), EX, "repeats with a positive mean difference")
    assert ((_ef[a] - _ef[b]).mean(1) > 0).sum() == rec["splits_positive"], key
_cv = {r["key"]: r["value"] for r in ROWS}
assert (_st["ALPHA"], _st["W_FULL"], _st["TAU"], _st["KEEP"]) == (_cv["c_alpha"], _cv["c_wfull"], _cv["c_tau"], _cv["c_keep"]), "AABC settings differ from the pipeline"
assert _st["INNER"] == _cv["c_inner"]
_gc = _ex["g_chosen"]["D0_I"]
put("ext_d0_g_edge_folds", int(_gc[str(_st["G_GRID"][0])]), EX, "$.g_chosen.D0_I['0.8'] (lowest grid value)")
put("ext_d0_g_next_folds", int(_gc[str(_st["G_GRID"][1])]), EX, "$.g_chosen.D0_I['0.85']")
assert sum(_gc.values()) == _st["N_REP"] * _st["N_FOLD"]
_nm = load(EXN)["null_check"]["mean_over_seeds"]
put("ext_null_lo", min(_nm.values()), EXN, "min over arms of the mean over synthetic targets")
put("ext_null_hi", max(_nm.values()), EXN, "max over arms of the mean over synthetic targets")
put("ext_null_seeds", len([k for k in load(EXN)["null_check"] if k != "mean_over_seeds"]), EXN, "synthetic targets")
_pt = [v for k, arms in load(EXN)["null_check"].items() if k != "mean_over_seeds" for v in arms.values()]
put("ext_null_tgt_lo", min(_pt), EXN, "min over synthetic targets and arms (one repeat of 10 folds each: run_null calls run_rep(0))")
put("ext_null_tgt_hi", max(_pt), EXN, "max over synthetic targets and arms")
EXD = "EXT_AABC_DRYRUN_SYNTHETIC_1119.json"
assert load(EXD)["info"]["mode"].startswith("synthetic")
put("ext_dry_lo", min(load(EXD)["mean_perfold_r"].values()), EXD, "min over arms, full analysis with one synthetic target")
put("ext_dry_hi", max(load(EXD)["mean_perfold_r"].values()), EXD, "max over arms, full analysis with one synthetic target")
take("ext_g_second", EX, "settings", "G_GRID", 1)
put("ext_d0_g_cg_folds", int(_gc[str(_cv["c_g"])]), EX, "$.g_chosen.D0_I at the benchmark's fixed shrinkage")
take("ns_d1_vs_fbert_lo", FH, "paired_vs_FBERT_best", "D1", "ci95", 0, nd=3)
take("ns_d1_vs_fbert_hi", FH, "paired_vs_FBERT_best", "D1", "ci95", 1, nd=3)
take("ns_d0_vs_fbert_d", FH, "paired_vs_FBERT_best", "D0", "diff", nd=3)
take("ns_d0_vs_fbert_lo", FH, "paired_vs_FBERT_best", "D0", "ci95", 0, nd=3)
take("ns_d0_vs_fbert_hi", FH, "paired_vs_FBERT_best", "D0", "ci95", 1, nd=3)
assert abs(dig(FH, ["holdout_r", "D1"]) - dig(NS, ["holdout_r", "D1"])) < 5e-4

# ---------------------------------------------------------------- foundation models, same folds
FS, FX, RV = "FBERT_SAMEFOLDS_1117.json", "FBERT_SIXMOD_1117.json", "REVIEW_CORRECTED_T_1118.json"
for k, a in {"fs_plain": "B_full", "fs_fbert8": "A8P1_full", "fs_fbert1": "A1P1_full", "fs_teacher": "C_full",
             "fs_d0": "D0_full", "fs_d1": "D1_full"}.items():
    take(k, FS, "arms", a, "mean", nd=4)
take("fs_seeds", FS, "n_seeds")
take("rest_d0_fbert_d", RV, "from_per_seed", "rest_D0_vs_FBERT8M", "diff")
take("rest_d0_fbert_t", RV, "from_per_seed", "rest_D0_vs_FBERT8M", "corrected_t")
take("rest_d0_fbert_p", RV, "from_per_seed", "rest_D0_vs_FBERT8M", "p_two_sided")
take("rest_d0_teacher_d", RV, "from_per_seed", "rest_D0_vs_FC035_teacher", "diff")
take("rest_d0_teacher_t", RV, "from_per_seed", "rest_D0_vs_FC035_teacher", "corrected_t")
for k, a in {"fx_plain": "B6_eq", "fx_teacher": "C6_eq", "fx_fbert_cat": "A8cat", "fx_d0": "D06_eq", "fx_d1": "D16_eq"}.items():
    take(k, FX, "arms", a, "mean", nd=4)
ps = load(FX)["per_seed"]
for k, (a, b) in {"six_d1_fbert": ("D16_eq", "A8cat"), "six_d0_fbert": ("D06_eq", "A8cat")}.items():
    dd = np.array(ps[a]) - np.array(ps[b])
    tn = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
    tc = tn / seed_factor(len(dd))
    put(f"{k}_d", round(float(dd.mean()), 4), FX, f"per_seed {a} - {b}")
    put(f"{k}_wins", int((dd > 0).sum()), FX, f"seeds {a} > {b}")
    put(f"{k}_t", round(float(tc), 2), FX, f"approximate NB t from per-seed scores (factor {seed_factor(len(dd)):.2f})")
    put(f"{k}_p", float(f"{2 * stats.t.sf(abs(tc), len(dd) - 1):.2g}"), FX, "two-sided p")
put("seed_factor_60", round(seed_factor(60), 2), FX, "sqrt((1/(kJ)+1/(k-1)) kJ), k=10, J=60")
put("seed_factor_20", round(seed_factor(20), 2), FX, "sqrt((1/(kJ)+1/(k-1)) kJ), k=10, J=20")

AO, AB, AD, NY = "FBERT_AOMIC_1117.json", "FBERT_ABIDE_1117.json", "FBERT_ADHD_1117.json", "FBERT_ADHD_NYU_WITHIN_1117.json"
take("aomic_n", AO, "n")
take("aomic_d0", AO, "arms", "D0", "mean", nd=4)
take("aomic_fbert", AO, "arms", "A8", "mean", nd=4)
take("aomic_teacher", AO, "arms", "C", "mean", nd=4)
take("aomic_d", AO, "PRIMARY_D0_vs_A8", "mean_diff", nd=3)
take("aomic_wins", AO, "PRIMARY_D0_vs_A8", "positive")
take("aomic_of", AO, "PRIMARY_D0_vs_A8", "of")
put("aomic_t", round(dig(AO, ["PRIMARY_D0_vs_A8", "t"]) / seed_factor(20), 2), AO, "seed t / seed_factor(20)")
take("abide_n", AB, "n_dx")
take("abide_nfiq", AB, "n_fiq")
take("abide_sites", AB, "n_sites")
take("abide_d0", AB, "tasks", "dx", "arms", "D0", "mean", nd=3)
take("abide_fbert", AB, "tasks", "dx", "arms", "A8", "mean", nd=3)
take("abide_d", AB, "tasks", "dx", "D0_vs_A8", "mean_diff", nd=3)
take("abide_wins", AB, "tasks", "dx", "D0_vs_A8", "positive")
put("abide_t", round(dig(AB, ["tasks", "dx", "D0_vs_A8", "t"]) / seed_factor(20), 2), AB, "seed t / seed_factor(20)")
take("abide_fiq_d0", AB, "tasks", "fiq", "arms", "D0", "mean", nd=3)
take("abide_fiq_fbert", AB, "tasks", "fiq", "arms", "A8", "mean", nd=3)
take("abide_fiq_d", AB, "tasks", "fiq", "D0_vs_A8", "mean_diff", nd=3)
take("abide_fiq_wins", AB, "tasks", "fiq", "D0_vs_A8", "positive")
put("abide_fiq_t", round(dig(AB, ["tasks", "fiq", "D0_vs_A8", "t"]) / seed_factor(20), 2), AB, "seed t / seed_factor(20)")
take("adhd_n", AD, "n_dx")
take("adhd_niq", AD, "n_iq")
put("adhd_sites", len(load(AD)["sites"]), AD, "len($.sites)")
take("adhd_d0", AD, "tasks", "dx", "arms", "D0", nd=3)
take("adhd_fbert", AD, "tasks", "dx", "arms", "A8", nd=3)
take("adhd_d", AD, "tasks", "dx", "D0_vs_A8", "diff", nd=3)
take("adhd_lo", AD, "tasks", "dx", "D0_vs_A8", "ci95", 0, nd=3)
take("adhd_hi", AD, "tasks", "dx", "D0_vs_A8", "ci95", 1, nd=3)
take("adhd_p", AD, "tasks", "dx", "D0_vs_A8", "p_gt_0", nd=2)
take("adhd_iq_d0", AD, "tasks", "iq", "arms", "D0", nd=3)
take("adhd_iq_fbert", AD, "tasks", "iq", "arms", "A8", nd=3)
put("adhd_iq_d", round(dig(AD, ["tasks", "iq", "arms", "D0"]) - dig(AD, ["tasks", "iq", "arms", "A8"]), 3), AD, "D0 - A8 IQ r")
take("adhd_iq_fbert1", AD, "tasks", "iq", "arms", "A1", nd=3)
take("nyu_n", NY, "n")
take("nyu_dx_fbert", NY, "dx", "A8", nd=3)
take("nyu_dx_d0", NY, "dx", "D0", nd=3)
take("nyu_dx_d0_ahead", NY, "dx", "D0_ahead_repeats")
take("bjepa_r", "FBERT_BJEPA_1117.json", "BrainJEPA", nd=3)
take("ra_raw_conditioning_without_filter_d", "REVIEW_ARMS_1119.json", "raw", "NB", "conditioning_without_filter", "diff_mean_perfold_r")
take("ra_raw_conditioning_without_filter_t", "REVIEW_ARMS_1119.json", "raw", "NB", "conditioning_without_filter", "corrected_t")
take("six_d0_g_top_folds", FX, "g_selected_counts", "D06_eq", "0.98")
put("six_d0_g_folds", sum(dig(FX, ["g_selected_counts", "D06_eq"]).values()), FX, "sum of $.g_selected_counts.D06_eq")
put("six_d0_plain_pooled_d", round(dig(FX, ["arms", "D06_eq", "mean"]) - dig(FX, ["arms", "B6_eq", "mean"]), 4), FX, "pooled per split: D06_eq - B6_eq")

# ---------------------------------------------------------------- exact fold-level tests (review, FOLDNB_*_1119)
# Each FOLDNB file holds every arm's per-fold scores from a rerun (code/fbert/fold_scores.py) of the original
# script; code/fbert/fold_nb.py wrote the corrected t. Here the t is recomputed from the stored folds and checked,
# and the rerun is checked against the original run's stored means before any value is used.
FE_PAIRS = {
    "SAMEFOLDS": ("rest", {"D0_full_vs_A8P1_full": "d0_fbert", "D0_full_vs_C_full": "d0_teacher", "C_full_vs_A8P1_full": "teacher_fbert",
                           "D1_full_vs_A8P1_full": "d1_fbert", "D1_full_vs_D0_full": "d1_d0", "B_full_vs_A8P1_full": "plain_fbert",
                           "D0_full_vs_B_full": "d0_plain", "D0_full_vs_A1P1_full": "d0_fbert1"}),
    "SIXMOD": ("six", {"D06_eq_vs_A8cat": "d0_fbert", "D16_eq_vs_A8cat": "d1_fbert", "C6_eq_vs_A8cat": "teacher_fbert",
                       "D16_eq_vs_D06_eq": "d1_d0", "D06_eq_vs_C6_eq": "d0_teacher", "D06_eq_vs_B6_eq": "d0_plain",
                       "D16_eq_vs_B6_eq": "d1_plain"}),
    "AOMIC": ("aomic", {"D0_vs_A8": "d0_fbert", "D1_vs_A8": "d1_fbert", "C_vs_A8": "teacher_fbert", "D0_vs_C": "d0_teacher", "D1_vs_D0": "d1_d0"}),
    "ABIDE_DX": ("abide", {"D0_vs_A8": "d0_fbert", "D1_vs_A8": "d1_fbert", "C_vs_A8": "teacher_fbert", "D0_vs_C": "d0_teacher", "D1_vs_D0": "d1_d0"}),
    "ABIDE_FIQ": ("abide_fiq", {"D0_vs_A8": "d0_fbert", "D1_vs_A8": "d1_fbert", "C_vs_A8": "teacher_fbert", "D0_vs_C": "d0_teacher", "D1_vs_D0": "d1_d0"}),
}
FE_ARMS = {
    "SAMEFOLDS": {"D0_full": "d0", "D1_full": "d1", "C_full": "teacher", "A8P1_full": "fbert8", "A1P1_full": "fbert1", "B_full": "plain"},
    "SIXMOD": {"D06_eq": "d0", "D16_eq": "d1", "C6_eq": "teacher", "A8cat": "fbert_cat", "B6_eq": "plain"},
    "AOMIC": {"D0": "d0", "D1": "d1", "C": "teacher", "A8": "fbert"},
    "ABIDE_DX": {"D0": "d0", "D1": "d1", "C": "teacher", "A8": "fbert"},
    "ABIDE_FIQ": {"D0": "d0", "D1": "d1", "C": "teacher", "A8": "fbert"},
}
# the rerun must reproduce the original run's stored means (per-fold mean for HCP; pooled r or AUC per repeat elsewhere)
FE_REPRO = {
    "SAMEFOLDS": ("FBERT_SAMEFOLDS_FOLDS_1119.json", "FBERT_SAMEFOLDS_1117.json", [("arms", a, "mean") for a in ["B_full", "A8P1_full", "A1P1_full", "C_full", "D0_full", "D1_full"]]),
    "SIXMOD": ("FBERT_SIXMOD_FOLDS_1119.json", "FBERT_SIXMOD_1117.json", [("arms", a, "mean") for a in ["B6_eq", "C6_eq", "A8cat", "D06_eq", "D16_eq"]]),
    "AOMIC": ("FBERT_AOMIC_FOLDS_1119.json", "FBERT_AOMIC_1117.json", [("arms", a, "mean") for a in ["D0", "D1", "C", "A8"]]),
    "ABIDE_DX": ("FBERT_ABIDE_FOLDS_1119.json", "FBERT_ABIDE_1117.json", [("tasks", "dx", "arms", a, "mean") for a in ["D0", "D1", "C", "A8"]]),
    "ABIDE_FIQ": ("FBERT_ABIDE_FOLDS_1119.json", "FBERT_ABIDE_1117.json", [("tasks", "fiq", "arms", a, "mean") for a in ["D0", "D1", "C", "A8"]]),
}
for tag, (pre, pairs) in FE_PAIRS.items():
    F = f"FOLDNB_{tag}_1119.json"
    rerun, orig, paths = FE_REPRO[tag]
    for pth in paths:
        a, b = dig(rerun, list(pth)), dig(orig, list(pth))
        assert abs(a - b) < 5e-4, (tag, pth, a, b)
    R = load(F)
    # the original runs stored pooled r (or AUC) per split; the paper's comparisons use the benchmark's metric,
    # the mean of per-fold scores, which is also what the corrected t tests
    for arm, short in FE_ARMS[tag].items():
        v = R["per_fold"][arm]
        m = float(np.nanmean(np.array([[np.nan if x is None else x for x in row] for row in v], dtype=float)))
        assert abs(round(m, 4) - R["mean_perfold"][arm]) < 1.1e-4, (tag, arm)
        put(f"fe_{pre}_m_{short}", round(m, 4), F, f"mean of $.per_fold.{arm}")
    ratio = R["test_train_ratio"]
    put(f"fe_{pre}_ratio", ratio, F, "$.test_train_ratio (" + R["ratio_source"] + ")")
    put(f"fe_{pre}_splits", R["splits"], F, "$.splits")
    for pair, short in pairs.items():
        a, b = pair.split("_vs_")
        d = (np.array(R["per_fold"][a], dtype=float) - np.array(R["per_fold"][b], dtype=float)).ravel()
        d = d[np.isfinite(d)]
        tc = d.mean() / math.sqrt((1.0 / len(d) + ratio) * d.var(ddof=1))
        rec = R["NB"][pair]
        assert abs(round(float(tc), 2) - rec["corrected_t"]) < 0.011 and abs(d.mean() - rec["diff"]) < 6e-5, (tag, pair, tc, rec)
        k = f"fe_{pre}_{short}"
        put(k + "_d", round(float(d.mean()), 4), F, f"$.per_fold {a} - {b}, mean over {len(d)} folds")
        put(k + "_t", round(float(tc), 2), F, f"corrected t recomputed from $.per_fold (ratio {ratio})")
        put(k + "_p", float(f"{2 * stats.t.sf(abs(tc), len(d) - 1):.2g}"), F, "two-sided p, df = folds - 1")
        put(k + "_pos", rec["splits_positive"], F, f"$.NB.{pair}.splits_positive")
        put(k + "_folds", len(d), F, f"finite paired folds {a}, {b}")


# short scan (first SHORT_T time points of one resting run): same rerun, same test, pairs not in fold_nb.py's list
_R = load("FOLDNB_SAMEFOLDS_1119.json")
_pu = load(FS)["fbert_probe_used"]     # the probe each F-BERT arm used (the better one, as pre-registered)
for a_, b_, short in [("D0_short", f"A8{_pu['A8_short']}_short", "d0_fbert"), ("D0_short", f"A1{_pu['A1_short']}_short", "d0_fbert1"), ("B_short", f"A8{_pu['A8_short']}_short", "plain_fbert"),
                      ("D0_short", "B_short", "d0_plain")]:
    d = (np.array(_R["per_fold"][a_], dtype=float) - np.array(_R["per_fold"][b_], dtype=float)).ravel()
    d = d[np.isfinite(d)]
    tc = d.mean() / math.sqrt((1.0 / len(d) + _R["test_train_ratio"]) * d.var(ddof=1))
    put(f"fe_short_{short}_d", round(float(d.mean()), 4), "FOLDNB_SAMEFOLDS_1119.json", f"$.per_fold {a_} - {b_}")
    put(f"fe_short_{short}_t", round(float(tc), 2), "FOLDNB_SAMEFOLDS_1119.json", f"corrected t from $.per_fold {a_}, {b_}")
for arm, short in [("D0_short", "d0"), (f"A8{_pu['A8_short']}_short", "fbert8"), (f"A1{_pu['A1_short']}_short", "fbert1"), ("B_short", "plain")]:
    put(f"fe_short_m_{short}", _R["mean_perfold"][arm], "FOLDNB_SAMEFOLDS_1119.json", f"$.mean_perfold.{arm}")


# ---------------------------------------------------------------- review arms (post hoc, REVIEW_ARMS_1119)
RA = "REVIEW_ARMS_1119.json"
assert load(RA)["sanity"]["reproduces_METRIC_CHECK_COV753"]
for a in ["CANON", "NOFILT", "FULLW1", "PARTW0", "PARTNA_W0", "FULL_FILT", "CBIG6", "CBIG6_HKH", "CBIG6_FILT", "CANON_EQ", "CANON_IND"]:
    take(f"ra_{a.lower()}_r", RA, "resid", "mean_perfold_r", a)
for k in ["CANON_vs_CBIG6", "NOFILT_vs_CBIG6", "filter_on_conditioned", "filter_on_full", "conditioning_without_filter",
          "conditioning_with_filter", "filter_on_benchmark_kernel", "benchmark_kernel_normalisation", "FULLW1_vs_CBIG6",
          "partial_step", "partial_without_power", "power_step", "mix_step", "language_weight", "inductive_filter_vs_none",
          "transductive_vs_inductive_filter", "CANON_IND_vs_CBIG6", "CANON_EQ_vs_CBIG6"]:
    kk = "ra_" + k.lower()
    take(kk + "_d", RA, "resid", "NB", k, "diff_mean_perfold_r")
    take(kk + "_t", RA, "resid", "NB", k, "corrected_t")
    take(kk + "_p", RA, "resid", "NB", k, "p_two_sided")
_pf = load(RA)["resid"]["per_fold_r"]
_inter = (np.array(_pf["CANON"]) - np.array(_pf["NOFILT"])) - (np.array(_pf["FULL_FILT"]) - np.array(_pf["FULLW1"]))
_d, _t, _p, _m = nb(_inter, np.zeros_like(_inter), load(RA)["test_train_ratio"])
put("ra_interaction_d", round(_d, 4), RA, "(CANON-NOFILT)-(FULL_FILT-FULLW1) per fold")
put("ra_interaction_t", round(_t, 2), RA, "NB t of the interaction")
put("ra_interaction_p", float(f"{_p:.2g}"), RA, "NB p of the interaction")
put("ra_ind_lam_interior", load(RA)["resid"]["lambda_chosen"]["CANON_IND"].get("0.001", 0), RA, "$.resid.lambda_chosen.CANON_IND['0.001']")
assert abs(dig(RA, ["resid", "mean_perfold_r", "FULLW1"]) - dig("ADDITIVE_CHAIN_RESIDUALISED_1118.json", ["fold_mean", "full_w1"])) < 1e-4
assert abs(dig(RA, ["resid", "mean_perfold_r", "PARTW0"]) - dig("ADDITIVE_CHAIN_RESIDUALISED_1118.json", ["fold_mean", "partial_w0"])) < 1e-4

# ---------------------------------------------------------------- disclosures requested by the review
C233 = "CANON_HOLDOUT_233_1117.json"
put("c233_d1", dig(C233, ["arms", "D1_benchmark_protocol", "r"]), C233, "$.arms.D1_benchmark_protocol.r")
put("c233_d0", dig(C233, ["arms", "D0_strict_inductive", "r"]), C233, "$.arms.D0_strict_inductive.r")
_m = re.search(r"moved ([0-9.]+) -> ([0-9.]+)", load(C233)["correction"])
put("c233_before_grid_fix", float(_m.group(1)), C233, "$.correction (value before the ridge grid was widened)")
_hid = load(C233)["holdout_ids"]   # a list here; {"removed", "n"} in the public bundle
put("c233_n", len(_hid) if isinstance(_hid, list) else _hid["n"], C233, "len($.holdout_ids)")
# Earlier use of subjects outside the benchmark's 753 (a superset of the 263), before the pre-registered run.
# held-out: they were test subjects whose target values scored predictions; train: their target values entered
# training in development analyses (larger cohorts, an intercept prior); features: unlabelled features only.
_H263 = set(map(str, load(NS)["holdout_ids"])) if isinstance(load(NS)["holdout_ids"], list) else None
PRIOR_ID = ["FBERT_HOLDOUT_1117.json", "FBERT_HOLDOUT_HYB_1117.json", "FBERT_JOINT_HOLDOUT_F_1117.json", "FBERT_MULTIATLAS_1117.json",
            "FBERT_S_MOTOR_HOLDOUT_1117.json", "FBERT_T_HOLDOUT_1117.json", "CANON_HOLDOUT_233_1117.json", "HO263.json", "HO263B.json",
            "LAMRULE.json", "MOTORHO.json", "RELHO.json", "RELHO2.json", "THREEHO.json"]
PRIOR_HELDOUT = ["CLIP_HOLDOUT_VALID_1119.json", "RANK_HOLDOUT_1118B.json", "FBERT_HOLDOUT_FAMILY_1117.json",
                 "EXTRACTION_PAIR_FOLD_INTERNAL_HOLDOUT_1117.json", "MOTOR_DROP_HOLDOUT_233_1117.json", "THREE_EXTRACTION_HOLDOUT_233_1117.json",
                 "RELATIONAL_REPEATED_HOLDOUT_1117.json", "RELATIONAL_BAGGED_SUBJECT_BOOTSTRAP_1117.json", "RELBAG.json", "RELREP.json",
                 "MOTORREP.json", "BCHO.json", "SEVENTH_TASK_RELATIONAL_1117.json", "RELATIONAL_BEST_SUBSET_1117.json"]
PRIOR_TRAIN = ["SIX961_NESTED_1117.json", "SIX961_WIDE_GRID_1117.json", "SIX_MODALITY_961_1117.json", "SIX974_a.json",
               "POPULATION_SCALING_1117.json", "POP_LEAKCONTROL_a.json", "POPULATION_LEAK_CONTROL_1117.json", "SIX961_LEAK_CONTROL_1117.json",
               "RS_AT_THEIR_COHORT_SIZE_1117.json", "EXTRA_SUBJECTS_1118B.json"]
PRIOR_FEAT = ["CKA961_b.json", "CKA_SIX_KERNELS_961_1117.json", "CKA_2X2_FILTER_G_1117.json", "CKA_SWEEP_G.json", "CKA_2x2.json"]
# early development (11-14, 11-15): cross-validated HCP cohorts larger than the 753 (n = 974: the 728 of the 753 with features plus
# subjects outside it; n = 949 at another parcellation; n = 1003 and 992: all HCP subjects with other cognitive targets),
# found by cohort size; these sizes occur only in HCP files here (the cohort runs of other datasets have other sizes)
HCP_CV_N = {974, 949, 1003, 992}
PRIOR_CV = []
for _f in sorted(os.listdir(RES)):
    if _f.endswith(".json") and not _f.startswith("FBERT_"):
        try:
            _d = json.load(open(os.path.join(RES, _f)))
        except Exception:
            continue
        def _sizes(x, out, depth=0):        # cohort sizes stored at any depth
            if isinstance(x, dict) and depth < 6:
                for k, v in x.items():
                    if k in ("n", "n_subjects", "n_total", "n_cohort") and isinstance(v, int):
                        out.add(v)
                    _sizes(v, out, depth + 1)
            elif isinstance(x, list) and depth < 6:
                for v in x[:20]:
                    _sizes(v, out, depth + 1)
        _ns = set(); _sizes(_d, _ns)
        if _ns & HCP_CV_N:
            PRIOR_CV.append(_f)


def _ids(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, list) and v and isinstance(v[0], (str, int)) and re.search(r"id|subj|hold|sid", k, re.I):
                out.update(map(str, v))
            _ids(v, out)


for f in PRIOR_ID:
    _s = set(); _ids(load(f), _s)
    assert _H263 is None or len(_s & _H263) > 0, f      # (the public bundle has the identifier lists removed)
PRIOR_CV = [f for f in PRIOR_CV if f not in set(PRIOR_ID + PRIOR_HELDOUT + PRIOR_TRAIN + PRIOR_FEAT)]
for f in PRIOR_HELDOUT + PRIOR_TRAIN + PRIOR_FEAT + PRIOR_CV:
    load(f)
PRIOR = PRIOR_ID + PRIOR_HELDOUT
# a byte-identical copy of a file counts once (several analyses also have a raw and a summary file; both are counted)
_seen_sha, _dups = {}, []
for f in PRIOR_CV + PRIOR + PRIOR_TRAIN + PRIOR_FEAT:
    if _sha[f] in _seen_sha:
        _dups.append(f)
    _seen_sha.setdefault(_sha[f], f)
_u = lambda L: [f for f in L if f not in _dups]
put("n_prior_cv", len(_u(PRIOR_CV)), NS, "early cross-validated development cohorts larger than the 753: " + ", ".join(_u(PRIOR_CV)))
put("n_prior_heldout", len(_u(PRIOR)), NS, "earlier result files using subjects outside the 753 as held-out test subjects: " + ", ".join(_u(PRIOR)))
put("n_prior_train", len(_u(PRIOR_TRAIN)), NS, "earlier result files adding their target values to training (incl. the intercept prior): " + ", ".join(_u(PRIOR_TRAIN)))
put("n_prior_feat", len(_u(PRIOR_FEAT)), NS, "earlier result files using their unlabelled features only: " + ", ".join(_u(PRIOR_FEAT)))
put("n_prior_looks", sum(len(_u(L)) for L in (PRIOR_CV, PRIOR, PRIOR_TRAIN, PRIOR_FEAT)), NS, "all of the above, identical copies counted once")
PRIOR_USE = {**{f: "early CV cohort" for f in PRIOR_CV}, **{f: "held-out" for f in PRIOR},
             **{f: "labels in training" for f in PRIOR_TRAIN}, **{f: "features only" for f in PRIOR_FEAT}}
PRIOR_DUP = {f: _seen_sha[_sha[f]] for f in _dups}
PO = "POOL_OVERLAP_1119.json"
take("pool_n", PO, "pool_n")
take("pool_outside", PO, "pool_outside_753")
take("pool_new_in_pool", PO, "new_subjects_in_pool")
assert dig(PO, ["new_subjects"]) == dig(NS, ["n_holdout"]) and dig(PO, ["new_subjects_in_753"]) == 0
assert all(load(f).get("n") == dig(PO, ["pool_n"]) for f in ["PARTIAL_ONLY_DISCOVERY_1115.json", "MIXED_PARTIAL_BEST_1115.json", "FINAL_METHOD_1115.json"])
_fd = load("FAMILY_DETECTOR_SENSITIVITY_1117.json")["detector_sensitivity"]["above_within753_p90_the_control"]
put("kin_detect_pct", round(100 * _fd, 1), "FAMILY_DETECTOR_SENSITIVITY_1117.json", "100 x $.detector_sensitivity.above_within753_p90_the_control")
_gs = [float(x) for x in _re.search(rb"^GS = \[([^\]]+)\]", _eh, _re.M).group(1).decode().split(",")]
put("ns_d1_g", _gs[dig(NS, ["chosen", "D1", "kernel"])], NS, "GS[chosen.D1.kernel] (GS from eval_holdout.py)")
put("ns_d0_g", _gs[dig(NS, ["chosen", "D0", "kernel"])], NS, "GS[chosen.D0.kernel]")
put("ns_g_max", max(_gs), NS, "max GS")
put("ns_g_min", min(_gs), NS, "min GS")
take("ns_d1_lam", NS, "chosen", "D1", "lambda")
take("ns_d0_lam", NS, "chosen", "D0", "lambda")
take("ns_cbig_lam", NS, "chosen", "CBIG6", "lambda")
_ev = open(os.path.join(HERE, "..", "code", "fbert", "eval_samefolds.py"), "rb").read()
_lg = _re.search(rb"^LAMBDAS = np.logspace\((-?\d+), (-?\d+), (\d+)\)", _ev, _re.M)
put_code("ns_lam_floor", 10.0 ** int(_lg.group(1)), "LAMBDAS min", "code/fbert/eval_samefolds.py")
put_code("ns_lam_top", 10.0 ** int(_lg.group(2)), "LAMBDAS max", "code/fbert/eval_samefolds.py")
take("adhd_iq_lo", AD, "tasks", "iq", "D0_vs_A8", "ci95", 0, nd=3)
take("adhd_iq_hi", AD, "tasks", "iq", "D0_vs_A8", "ci95", 1, nd=3)
take("aomic_d1", AO, "arms", "D1", "mean", nd=4)
take("abide_d1", AB, "tasks", "dx", "arms", "D1", "mean", nd=3)
take("abide_teacher", AB, "tasks", "dx", "arms", "C", "mean", nd=3)
take("abide_fiq_d1", AB, "tasks", "fiq", "arms", "D1", "mean", nd=3)
take("abide_fiq_teacher", AB, "tasks", "fiq", "arms", "C", "mean", nd=3)
take("adhd_d1", AD, "tasks", "dx", "arms", "D1", nd=3)
take("adhd_teacher", AD, "tasks", "dx", "arms", "C", nd=3)
take("adhd_iq_d1", AD, "tasks", "iq", "arms", "D1", nd=3)
take("adhd_iq_teacher", AD, "tasks", "iq", "arms", "C", nd=3)
TT = "THREE_TARGETS_1116.json"
for tgt, key in [("varimax_er", "er"), ("varimax_satisf", "sat")]:
    put(f"other_{key}_chain", round(float(np.mean(dig(TT, [tgt, "v4"]))), 4), TT, f"mean($.{tgt}.v4)")
    put(f"other_{key}_plain", round(float(np.mean(dig(TT, [tgt, "baseline_full_FC"]))), 4), TT, f"mean($.{tgt}.baseline_full_FC)")
    _dd = np.array(dig(TT, [tgt, "v4"])) - np.array(dig(TT, [tgt, "baseline_full_FC"]))
    put(f"other_{key}_t", round(float(_dd.mean() / (_dd.std(ddof=1) / math.sqrt(len(_dd))) / seed_factor(len(_dd))), 2), TT,
        "approximate corrected t from per-split means (naive t / seed_factor)")
take("adhd_d1_fbert_lo", AD, "tasks", "dx", "D1_vs_A8", "ci95", 0, nd=3)
take("adhd_d1_fbert_hi", AD, "tasks", "dx", "D1_vs_A8", "ci95", 1, nd=3)
take("adhd_iq_d1_fbert_lo", AD, "tasks", "iq", "D1_vs_A8", "ci95", 0, nd=3)
take("adhd_iq_d1_fbert_hi", AD, "tasks", "iq", "D1_vs_A8", "ci95", 1, nd=3)
_fv_ = load("FBERT_VARIANTS_1117.json")
put("fbert_loss_recorded", round(_fv_["recorded_loss"]["8m"], 4), "FBERT_VARIANTS_1117.json", "$.recorded_loss.8m (the checkpoint's recorded training loss)")
put("fbert_loss_rebuilt", round(min(v["loss"] for v in _fv_["variants"] if v["model"] == "8m"), 3), "FBERT_VARIANTS_1117.json",
    "min loss over the rebuilt forward-pass variants, 8M (the variant used)")
if os.path.exists(_exp):
    _fv, _fs = json.load(open(_exp))["fbert_paper_hcp"], "results/_EXPORTED_PRIVATE_VALUES.json"
else:
    _fv, _fs = json.load(open(os.path.join(HERE, "..", "CANON_NUMBERS.json")))["their_table3"]["F-BERT-8M"]["r"], "CANON_NUMBERS.json"
ROWS.append({"key": "fbert_paper_hcp", "value": _fv, "source_file": _fs, "sha16": hashlib.sha256(open(os.path.join(HERE, "..", _fs) if _fs == "CANON_NUMBERS.json" else _exp, "rb").read()).hexdigest()[:16],
             "locator": "the F-BERT preprint's table, F-BERT-8M on HCP"}); SEEN.add("fbert_paper_hcp")

# ---------------------------------------------------------------- post hoc levers (Table S1)
LEV = []


def lever(key, label, f, rpath, dpath, tsrc, note):
    r = dig(f, rpath) if rpath else None
    d = dig(f, dpath)
    t = None
    if tsrc is not None:
        tf, tpath = tsrc
        t = abs(dig(tf, tpath)) * (1 if d >= 0 else -1)
    put(f"lev_{key}_d", round(d, 4), f, "$." + ".".join(map(str, dpath)))
    if t is not None:
        put(f"lev_{key}_t", round(t, 2), tsrc[0], "$." + ".".join(map(str, tsrc[1])) + " (sign of the difference)")
    LEV.append((key, label, t is not None, note))


lever("rel", "RELATIONAL task added at a fixed one-ninth share", RV, None,
      ["from_recorded_t", "REL_1of9_0.7382_vs_0.7357", "diff"],
      (RV, ["from_recorded_t", "REL_1of9_0.7382_vs_0.7357", "corrected_t"]), "fixed setting; four fold-internal rules failed")
take("clip_cert_r", "CLIP_COV_GATE_1118B.json", "cert_raw_value_kernel")
lever("clip", "edge clipping at 1.5 robust SD (covariate-matched)", "CLIP_COV_GATE_1118B.json", None,
      ["clip_fixed", "diff"], ("CLIP_COV_GATE_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "clip_fixed", "NB_t"]),
      "held-out test void (execution error)")
lever("rank", "Spearman (rank) kernel", "RANKLAM_1118B.json", None, ["rank_kernel", "diff"],
      ("RANKLAM_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "rank_kernel", "NB_t"]), "")
lever("winsor", "winsorised target (2.5 SD)", "TARGET_TRANSFORM_1118B.json", None, ["winsor_2.5", "diff"],
      ("TARGET_TRANSFORM_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "winsor_2.5", "NB_t"]), "scored against the transformed target")
lever("kta", "kernel-target alignment weights", "KTA_STRUCT_PRIOR_1118B.json", None, ["kta_power_1.0", "diff"], None,
      "40 seeds, no paired test")
lever("labelgram", "label-Gram kernel augmentation (weight 0.10)", "AUX_TARGET_1118B.json", None,
      ["arms_60seeds", "label_gram_b0.10", "diff"],
      ("AUX_TARGET_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "label_gram_b0.10", "NB_t"]), "")
lever("extra", "mean target of the labelled subjects outside the 753 as intercept prior", "EXTRA_SUBJECTS_1118B.json", None,
      ["extra_label_mean_intercept", "diff"],
      ("EXTRA_SUBJECTS_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "extra_label_mean_intercept", "NB_t"]), "")
lever("aux", "auxiliary composite target", "AUX_TARGET_1118B.json", None, ["arms_60seeds", "multi_task_composite", "diff"],
      ("AUX_TARGET_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "multi_task_composite", "NB_t"]), "")
lever("ranktarget", "rank-normalised target", "TARGET_TRANSFORM_1118B.json", None, ["rank_normal_target", "diff"],
      ("TARGET_TRANSFORM_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "rank_normal_target", "NB_t"]), "scored against the transformed target")
lever("alpha", "chain kernel replaced by the filtered kernel of eigenvalue-powered connectomes (power 0.35, weight 1)", "ALPHA_ON_TOP_1119.json", None,
      ["alpha_only", "diff"], ("ALPHA_ON_TOP_1119.json", ["alpha_only", "NB_t"]), "largest of four weights tried")
lever("alphamix", "the same kernel mixed into the chain at weight 0.10", "ALPHA_ON_TOP_1119.json", None, ["mix_0.10", "diff"],
      ("ALPHA_ON_TOP_1119.json", ["mix_0.10", "NB_t"]), "weights 0.25 and 0.50 also not significant")
lever("rankmix", "equal mix of the chain kernel and the Spearman kernel (hand-set)", "RANKLAM_1118B.json", None,
      ["certified_plus_rank_mix", "diff"], ("RANKLAM_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "certified_plus_rank_mix", "NB_t"]),
      "fixed weight chosen by hand")
lever("clipmix", "equal mix of the raw-edge and clipped-edge kernels (hand-set)", "CLIP_GATE_1118B.json", None,
      ["fixed_equal_mix", "diff"], ("CLIP_GATE_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "fixed_equal_mix", "NB_t"]),
      "fixed weight chosen by hand; raw target, no covariate matching")
lever("lambag", "averaging predictions over six ridge penalties", "RANKLAM_1118B.json", None, ["lambda_bag", "diff"],
      ("RANKLAM_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "lambda_bag", "NB_t"]), "")
_cc = dig("CLIP_COV_GATE_1118B.json", ["_NB_T_CORRECTION_1119", "corrected", "clip_fixed"])
put("nb_bug_factor", round(_cc["reported_t"] / _cc["NB_t"], 1), "CLIP_COV_GATE_1118B.json", "reported_t / NB_t, clip_fixed")
V = {r["key"]: r["value"] for r in ROWS}
pos_sig = sum(1 for k, *_ in LEV if V.get(f"lev_{k}_t") is not None and V[f"lev_{k}_t"] > 1.96)
put("lev_n", len(LEV), "RANKLAM_1118B.json", "count of rows in Table S1")
put("lev_pos_sig", pos_sig, "RANKLAM_1118B.json", "rows with corrected t > 1.96")
neg_sig = sum(1 for k, *_ in LEV if V.get(f"lev_{k}_t") is not None and V[f"lev_{k}_t"] < -1.96)
put("lev_neg_sig", neg_sig, "RANKLAM_1118B.json", "rows with corrected t < -1.96")

# ---------------------------------------------------------------- pre-registered contrasts (Table S2)
# group: single (one decision), multi (a rule per cell), secondary (declared with the same rule as the
# primary), status (registered, but voided, never executed, or a diagnostic with a reading rule)
PR = []


def ROWS_BY_KEY(k):
    return next(r["value"] for r in ROWS if r["key"] == k)


def pr_row(key, label, met, rule, observed_keys, group="single"):
    put(f"pr_{key}_met", bool(met), observed_keys[0][0] if observed_keys else NS, "decision rule applied to stored values")
    PR.append((key, label, rule, group))


def pr_seeds(key, label, f, wins_path, of_path, need, group="single"):
    w, of = int(dig(f, wins_path)), int(dig(f, of_path))
    put(f"pr_{key}_wins", w, f, "$." + ".".join(map(str, wins_path)))
    put(f"pr_{key}_of", of, f, "$." + ".".join(map(str, of_path)))
    put(f"pr_{key}_need", need, f, "pre-registered threshold")
    pr_row(key, label, w >= need, "seeds", [(f,)], group)


def pr_p(key, label, f, base, extra_ok=True):
    pv = dig(f, base + ["p_gt_0"])
    put(f"pr_{key}_p", round(pv, 3), f, "$." + ".".join(map(str, base + ["p_gt_0"])))
    put(f"pr_{key}_lo", round(dig(f, base + ["ci95", 0]), 3), f, "$." + ".".join(map(str, base + ["ci95", 0])))
    put(f"pr_{key}_hi", round(dig(f, base + ["ci95", 1]), 3), f, "$." + ".".join(map(str, base + ["ci95", 1])))
    pr_row(key, label, pv >= 0.95 and extra_ok, "p", [(f,)])


def pr_ci(key, label, f, base):
    lo = dig(f, base + ["ci95", 0])
    put(f"pr_{key}_lo", round(lo, 3), f, "$." + ".".join(map(str, base + ["ci95", 0])))
    put(f"pr_{key}_hi", round(dig(f, base + ["ci95", 1]), 3), f, "$." + ".".join(map(str, base + ["ci95", 1])))
    pr_row(key, label, lo > 0, "ci", [(f,)])


def rule_ok(e):
    """an entry decided by bootstrap P (leave-one-site-out cells) or by repeats (>= 95 %)"""
    return (e["p_gt_0"] >= 0.95) if "p_gt_0" in e else (e["positive"] >= 0.95 * e["of"])


pr_seeds("rest", "HCP resting state, D0 vs F-BERT-8M (main contrast)", FS, ["contrasts", "full", "D0-A8", "positive"], ["contrasts", "full", "D0-A8", "of"], 57)
pr_seeds("a", "HCP six modalities, D0 vs F-BERT-8M (A)", FX, ["contrasts", "eq", "D0-A8", "positive"], ["contrasts", "eq", "D0-A8", "of"], 57)
pr_seeds("b", "six modalities, in-fold choice among eighteen kernels vs F-BERT-8M (B)", "FBERT_SIXMOD_PLUS_1117.json",
         ["contrasts", "PLUS-A8cat", "positive"], ["contrasts", "PLUS-A8cat", "of"], 57)
pr_seeds("c", "six modalities, removing a shared cross-modality component, vs F-BERT-8M (C)", "FBERT_SIXMOD_DECOR_1117.json",
         ["contrasts", "DECOR-A8DECOR", "positive"], ["contrasts", "DECOR-A8DECOR", "of"], 57)
_d1 = dig(FH, ["holdout_r", "D1"]); _fb = max(dig(FH, ["holdout_r", "A8cat"]), dig(FH, ["holdout_r", "A8F"]))
pr_p("d", "held-out subjects, D1 vs F-BERT-8M (D)", FH, ["primary_D1_vs_FBERT_best"], extra_ok=_d1 > _fb)
pr_p("e", "held-out subjects, half D1 and half F-BERT vs F-BERT-8M (E)", "FBERT_HOLDOUT_HYB_1117.json", ["primary_HYB_vs_FBERT_best"])
pr_p("f", "held-out intersection, average with a second extraction vs D1 (F; rule amended after both components' held-out r was known)",
     "FBERT_JOINT_HOLDOUT_F_1117.json", ["PRIMARY_PRED50_vs_ours_D1"])
pr_p("g", "held-out subjects, three-atlas average vs D1 (G)", "FBERT_MULTIATLAS_1117.json", ["part2_holdout", "PRIMARY_MA_vs_D1_450"])
pr_seeds("h", "AOMIC-ID1000, D0 vs F-BERT-8M (H)", AO, ["PRIMARY_D0_vs_A8", "positive"], ["PRIMARY_D0_vs_A8", "of"], 19)
pr_seeds("i", "HCP resting state, D0 vs Brain-JEPA (I)", "FBERT_BJEPA_1117.json", ["PRIMARY_D0_vs_BrainJEPA", "positive"], ["PRIMARY_D0_vs_BrainJEPA", "of"], 57)
pr_seeds("j", "ABIDE-I autism, D0 vs F-BERT-8M (J)", AB, ["tasks", "dx", "D0_vs_A8", "positive"], ["tasks", "dx", "D0_vs_A8", "of"], 19)
pr_p("l", "ADHD-200 diagnosis, D0 vs F-BERT-8M (L)", AD, ["tasks", "dx", "D0_vs_A8"])
pr_seeds("r", "dropping the motor task, D1 (R)", "FBERT_R_MOTOR_1117.json", ["PRIMARY_D1_nomotor_vs_six", "positive"], ["PRIMARY_D1_nomotor_vs_six", "of"], 57)
pr_p("s", "dropping the motor task, held-out subjects, D0 (S)", "FBERT_S_MOTOR_HOLDOUT_1117.json", ["PRIMARY_D0_nomotor_vs_six"])
pr_seeds("t", "adding the RELATIONAL task, D1, benchmark splits (T)", "FBERT_T_INFOLD_1117.json", ["PRIMARY_D1_rel_vs_six", "positive"], ["PRIMARY_D1_rel_vs_six", "of"], 57)
pr_p("t2", "adding the RELATIONAL task, D1, held-out subjects (T)", "FBERT_T_HOLDOUT_1117.json", ["COPRIMARY_D1_rel_vs_six"])
pr_ci("ns", "held-out subjects, D1 vs the benchmark kernel (this paper)", NS, ["primary_D1_vs_CBIG6"])
put("pr_ext_t", V_ext_t := ROWS_BY_KEY("ext_d0_cbig_t"), EX, "corrected t of the primary contrast")
pr_row("ext", "independent cohort (AABC), inductive conditioned kernel vs the benchmark kernel (this paper)",
       ROWS_BY_KEY("ext_d0_cbig_d") > 0 and V_ext_t >= ROWS_BY_KEY("ext_tcrit"), "t", [(EX,)])

# multi-cell contrasts
UN = "FBERT_UNION_SUMMARY_1117.json"
_k = [(c, v["U_vs_A8"][1], v["U_vs_A8"][2]) for c, v in load(UN)["cohorts"].items()]
_kmet = sum(1 for c, w, of in _k if w >= (57 if of == 60 else 19))
put("pr_k_cells", len(_k), UN, "cohorts")
put("pr_k_cells_met", _kmet, UN, "cohorts with U - A8 positive in the threshold number of repeats")
pr_row("k", "D0 and its pre-training target chosen in-fold vs F-BERT-8M, per cohort (K)", _kmet == len(_k), "cells", [(UN,)], "multi")
N_FILES = [("FBERT_SOTA_HCP_rest_1117.json", ["rest"]), ("FBERT_SOTA_HCP_six_1117.json", ["six"]),
           ("FBERT_SOTA_AOMIC_1117.json", ["ist"]), ("FBERT_SOTA_ABIDE_1117.json", ["dx", "fiq"]),
           ("FBERT_SOTA_ADHD_1117.json", ["dx", "iq"]), ("FBERT_SOTA_ADHD_NYU_1117.json", ["dx", "iq"])]
_ncells, PREV = [], {}
COH = {"FBERT_SOTA_HCP_rest_1117.json": "HCP_rest", "FBERT_SOTA_HCP_six_1117.json": "HCP_six", "FBERT_SOTA_AOMIC_1117.json": "aomic",
       "FBERT_SOTA_ABIDE_1117.json": "abide", "FBERT_SOTA_ADHD_1117.json": "adhd", "FBERT_SOTA_ADHD_NYU_1117.json": "adhd_nyu"}
for f, cells in N_FILES:
    for c in cells:
        _ncells.append(all(rule_ok(dig(f, ["cells", c, comp])) for comp in ("SOTA_vs_A8", "SOTA_vs_OURS")))
        PREV[(COH[f], c)] = max(dig(f, ["cells", c, "mean"]).values())
put("pr_n_cells", len(_ncells), "FBERT_SOTA_HCP_six_1117.json", "cells of Addendum N (ADHD-200 within NYU counted as two cells, diagnosis and IQ)")
put("pr_n_cells_met", sum(_ncells), "FBERT_SOTA_HCP_six_1117.json", "cells where the combined system beat both comparators")
pr_row("n", "combined in-fold system vs F-BERT-8M and D0, per cell (N)", all(_ncells), "cells", [("FBERT_SOTA_HCP_six_1117.json",)], "multi")
_ocells = []
for f, c, rule, coh in [("FBERT_SOTA_O_adhd_1117.json", "iq", "p", "adhd"), ("FBERT_SOTA_O_adhd_nyu_1117.json", "dx", "seeds", "adhd_nyu")]:
    ok = True
    for comp in ("SOTA_vs_A8", "SOTA_vs_A1", "SOTA_vs_OURS"):
        e = dig(f, ["cells", c, comp])
        ok &= (e["p_gt_0"] >= 0.95) if rule == "p" else (e["positive"] >= 19)
    _ocells.append(ok)
    PREV[(coh, c)] = max(PREV[(coh, c)], max(dig(f, ["cells", c, "mean"]).values()))
put("pr_o_cells", len(_ocells), "FBERT_SOTA_O_adhd_1117.json", "cells of Addendum O")
put("pr_o_cells_met", sum(_ocells), "FBERT_SOTA_O_adhd_1117.json", "cells won")
pr_row("o", "the two ADHD-200 cells, combined system vs both F-BERT models and D0, per cell (O)", all(_ocells), "cells", [("FBERT_SOTA_O_adhd_1117.json",)], "multi")
# P and Q: our own encoders (C-BERT) and combined systems with them in the pool. A cell is won only if the system beats
# F-BERT-8M and D0 by the cell's rule and every earlier arm (N, O); the earlier arms' means are an upper bar the system
# must clear before any test, so a cell whose mean is below it is not met.
for key, label, pat in [("p", "combined system with our own pre-trained encoder in the pool, per cell (P)", "FBERT_P_{}_k0.json"),
                        ("q1", "combined system with ensembles of encoders in the pool, per cell (Q1)", None),
                        ("q2", "combined system with a larger own encoder in the pool, per cell (Q2)", "FBERT_Q2_{}_k0.json"),
                        ("q3", "combined system with the largest own encoder in the pool, per cell (Q3)", "FBERT_Q3_{}_k0.json")]:
    files = ([pat.format(c) for c in ["HCP_rest", "HCP_six", "aomic", "abide", "adhd", "adhd_nyu"]] if pat else
             ["FBERT_Q1_aomic.json", "FBERT_Q1_abide.json", "FBERT_Q1_adhd.json", "FBERT_Q1_adhd_nyu.json"])
    cells_ok = []
    for f in files:
        coh = re.sub(r"^FBERT_(P|Q1|Q2|Q3)_|(_k0)?\.json$", "", f)
        for c, v in load(f)["cells"].items():
            beat_both = rule_ok(v["SOTA_vs_A8"]) and rule_ok(v["SOTA_vs_OURS"])
            above_prev = v["mean"]["SOTA"] > PREV[(coh, c)]
            assert not (beat_both and above_prev), (f, c, "would need a test against the earlier arm; not stored")
            cells_ok.append(False)
    put(f"pr_{key}_cells", len(cells_ok), files[0], "cells of the addendum")
    put(f"pr_{key}_cells_met", sum(cells_ok), files[0], "cells won (see the rule above)")
    pr_row(key, label, all(cells_ok), "cells", [(files[0],)], "multi")
# C-BERT alone against F-BERT-8M (the first half of P's primary): the files store means, not the per-repeat counts the rule needs
_cb = [(f, c, v["mean"]["CB"] > v["mean"]["A8"]) for f in ["FBERT_P_{}_k0.json".format(x) for x in ["HCP_rest", "HCP_six", "aomic", "abide", "adhd", "adhd_nyu"]]
       for c, v in load(f)["cells"].items()]
put("pr_cbert_cells", len(_cb), "FBERT_P_HCP_rest_k0.json", "cells of Addendum P")
put("pr_cbert_ahead", sum(1 for *_, a in _cb if a), "FBERT_P_HCP_rest_k0.json", "cells where C-BERT's mean exceeds F-BERT-8M's")
put("pr_cbert_behind", sum(1 for *_, a in _cb if not a), "FBERT_P_HCP_rest_k0.json", "cells where C-BERT's mean is below F-BERT-8M's (the every-cell rule fails)")
pr_row("cbert", "our own pre-trained encoder alone vs F-BERT-8M, per cell (P)", False, "cells", [("FBERT_P_HCP_rest_k0.json",)], "multi")

# secondary contrasts declared "same rule" in the main pre-registration (full scan) and the short-scan arm
SEC = [("full", "D0-A1", "HCP resting state, D0 vs F-BERT-1M"), ("full", "D1-A8", "HCP resting state, D1 vs F-BERT-8M"),
       ("full", "B-A8", "HCP resting state, plain FC vs F-BERT-8M"), ("full", "C-A8", "HCP resting state, powered FC vs F-BERT-8M"),
       ("short", "D0-A8", "short scan, D0 vs F-BERT-8M"), ("short", "D0-A1", "short scan, D0 vs F-BERT-1M"),
       ("short", "B-A8", "short scan, plain FC vs F-BERT-8M"),
       ("short", "C-A8", "short scan, powered FC vs F-BERT-8M"), ("short", "D0-B", "short scan, D0 vs plain FC")]
for i_, (scan, c, label) in enumerate(SEC):
    pr_seeds(f"sec{i_}", label, FS, ["contrasts", scan, c, "positive"], ["contrasts", scan, c, "of"], 57, "secondary")

# registered but not decided
for key, label, status, f in [("m", "what F-BERT's ADHD-200 within-site edge reflects (M, diagnostic with a reading rule)", "diagnostic", "FBERT_ADHD_MOTION_1117.json"),
                              ("clip", "clipping extreme edges, held-out subjects (separate pre-registration)", "voided", "CLIP_COV_GATE_1118B.json"),
                              ("chainho", "chain vs plain FC, held-out subjects (separate pre-registration)", "not executed", NS)]:
    PR.append((key, label, status, "status"))
_mr = load("FBERT_ADHD_MOTION_1117.json")["reading"]
put("pr_m_dvars_alone_ok", bool(_mr["dvars_alone_ge_0.60"]), "FBERT_ADHD_MOTION_1117.json", "$.reading.dvars_alone_ge_0.60")
put("pr_m_half_removed", bool(_mr["half_of_margin_removed"]), "FBERT_ADHD_MOTION_1117.json", "$.reading.half_of_margin_removed")
_mm = load("FBERT_ADHD_MOTION_1117.json")["mean"]
put("pr_m_dvars_auc", round(_mm["DV"], 3), "FBERT_ADHD_MOTION_1117.json", "$.mean.DV")
put("pr_m_best_temporal", round(max(v for k, v in _mm.items() if k in ("AC1", "AC1+D0", "XC1", "XC1+D0", "DFC_SD", "DFC_SD+D0")), 3),
    "FBERT_ADHD_MOTION_1117.json", "max of the closed-form temporal arms")
put("pr_m_fbert_auc", round(_mm["A8"], 3), "FBERT_ADHD_MOTION_1117.json", "$.mean.A8")
put("pr_m_margin", round(_mr["fbert_margin_raw"], 3), "FBERT_ADHD_MOTION_1117.json", "$.reading.fbert_margin_raw")
put("pr_m_margin_after", round(_mr["fbert_margin_after_dvars"], 3), "FBERT_ADHD_MOTION_1117.json", "$.reading.fbert_margin_after_dvars")
_temporal = max(v for k, v in _mm.items() if k in ("AC1", "AC1+D0", "XC1", "XC1+D0", "DFC_SD", "DFC_SD+D0"))
put("pr_m_temporal_reaches", bool(_temporal >= _mm["A8"]), "FBERT_ADHD_MOTION_1117.json", "best closed-form temporal arm >= F-BERT")

V = {r["key"]: r["value"] for r in ROWS}
single = [k for k, _, _, g in PR if g == "single"]
put("pr_n", len(single), NS, "single-decision rows of Table S2")
put("pr_met", sum(1 for k in single if V[f"pr_{k}_met"]), NS, "single-decision rows met")
put("pr_notmet", sum(1 for k in single if not V[f"pr_{k}_met"]), NS, "single-decision rows not met")
put("pr_met_by_repeats", sum(1 for k, _, r, g in PR if g == "single" and r == "seeds" and V[f"pr_{k}_met"]), NS, "met rows decided by counting repeats")
sec = [k for k, _, _, g in PR if g == "secondary"]
put("pr_sec_n", len(sec), FS, "secondary rows")
put("pr_sec_met", sum(1 for k in sec if V[f"pr_{k}_met"]), FS, "secondary rows met")
put("pr_multi_n", sum(1 for *_, g in PR if g == "multi"), NS, "multi-cell rows")

# D0 against F-BERT-8M over the eight settings of Table 4 (a descriptive count: the HCP settings share subjects)
V = {r["key"]: r["value"] for r in ROWS}
_FM = [("fe_rest_m_d0", "fe_rest_m_fbert8", V["fe_rest_d0_fbert_t"] > 1.96), ("fe_six_m_d0", "fe_six_m_fbert_cat", V["fe_six_d0_fbert_t"] > 1.96),
       ("ns_d0_r", "ns_fbert_r", V["ns_d0_vs_fbert_lo"] > 0), ("fe_aomic_m_d0", "fe_aomic_m_fbert", V["fe_aomic_d0_fbert_t"] > 1.96),
       ("fe_abide_m_d0", "fe_abide_m_fbert", V["fe_abide_d0_fbert_t"] > 1.96), ("fe_abide_fiq_m_d0", "fe_abide_fiq_m_fbert", V["fe_abide_fiq_d0_fbert_t"] > 1.96),
       ("adhd_d0", "adhd_fbert", V["adhd_lo"] > 0), ("adhd_iq_d0", "adhd_iq_fbert", V["adhd_iq_lo"] > 0)]
put("fm_n", len(_FM), FH, "rows of Table 4")
put("fm_ahead", sum(1 for a_, b_, _ in _FM if V[a_] > V[b_]), FH, "rows of Table 4 where D0's score exceeds F-BERT-8M's")
put("fm_sig", sum(1 for a_, b_, sg in _FM if V[a_] > V[b_] and sg), FH, "rows where D0 is ahead with corrected t > 1.96 or a 95% interval above zero")

os.makedirs(os.path.join(HERE, "ledger"), exist_ok=True)
json.dump({"note": "built by build_numbers.py from ../results; do not edit by hand", "rows": ROWS},
          open(os.path.join(HERE, "ledger", "numbers_index.json"), "w"), indent=1)
SCRIPTS = {
    "METRIC_CHECK_COV753_1118.json": "code/fbert/metric_check.py (VARIANTS=1)", "METRIC_CHECK_COV753_REST_1118.json": "code/fbert/metric_check.py (REST_ONLY=1)",
    "METRIC_CHECK_1118.json": "code/fbert/metric_check.py (original ridge grid)", "NESTED_MATCHED_1118.json": "code/fbert/nested_matched.py",
    "REVIEW_ARMS_1119.json": "code/fbert/review_arms.py", "CHAIN_VS_OOI_KERNEL_263_1119.json": "code/fbert/eval_holdout_chain.py",
    "FBERT_HOLDOUT_1117.json": "code/fbert/eval_holdout.py", "FBERT_SAMEFOLDS_1117.json": "code/fbert/eval_samefolds.py",
    "FBERT_SIXMOD_1117.json": "code/fbert/eval_sixmod.py", "FBERT_AOMIC_1117.json": "code/fbert/aomic_eval.py",
    "FBERT_ABIDE_1117.json": "code/fbert/abide_eval.py", "FBERT_ADHD_1117.json": "code/fbert/adhd_eval.py",
    "FBERT_ADHD_NYU_WITHIN_1117.json": "code/fbert/adhd_within.py", "FBERT_BJEPA_1117.json": "code/fbert/eval_bjepa.py",
    "FBERT_SIXMOD_PLUS_1117.json": "code/fbert/eval_sixmod_plus.py", "FBERT_SIXMOD_DECOR_1117.json": "code/fbert/eval_sixmod_decor.py",
    "FBERT_HOLDOUT_HYB_1117.json": "code/fbert/eval_holdout_hyb.py",
    "FBERT_MULTIATLAS_1117.json": "code/fbert/eval_multiatlas.py", "FBERT_R_MOTOR_1117.json": "code/fbert/motor_drop.py",
    "FBERT_S_MOTOR_HOLDOUT_1117.json": "code/fbert/motor_holdout.py", "FBERT_T_INFOLD_1117.json": "code/fbert/rel_eval.py (infold)",
    "FBERT_T_HOLDOUT_1117.json": "code/fbert/rel_eval.py (holdout)", "FBERT_SOTA_HCP_rest_1117.json": "code/fbert/sota_hcp.py",
    "FBERT_SOTA_HCP_six_1117.json": "code/fbert/sota_hcp.py", "FBERT_SOTA_AOMIC_1117.json": "code/fbert/sota_single.py",
    "FBERT_SOTA_ABIDE_1117.json": "code/fbert/sota_single.py", "FBERT_SOTA_ADHD_1117.json": "code/fbert/sota_single.py",
    "FBERT_SOTA_ADHD_NYU_1117.json": "code/fbert/sota_single.py", "FBERT_SOTA_O_adhd_1117.json": "code/fbert/sota_single.py",
    "FBERT_SOTA_O_adhd_nyu_1117.json": "code/fbert/sota_single.py", "REVIEW_CORRECTED_T_1118.json": "code/fbert/corrected_t_review.py",
    "CLIP_COV_GATE_1118B.json": "code/clip_cov_gate.py", "RANKLAM_1118B.json": "code/rank_lam.py", "CLIP_GATE_1118B.json": "code/clip_gate.py", "POOL_OVERLAP_1119.json": "code/fbert/pool_overlap.py",
    "CLIP_HOLDOUT_VALID_1119.json": "code/clip_holdout_valid.py", "TARGET_TRANSFORM_1118B.json": "code/target_transform.py",
    "KTA_STRUCT_PRIOR_1118B.json": "code/kta_struct_prior.py", "AUX_TARGET_1118B.json": "code/aux_target.py",
    "EXTRA_SUBJECTS_1118B.json": "code/extra_subjects.py", "ALPHA_ON_TOP_1119.json": "code/alpha_on_top.py",
    "EXT_AABC_1119.json": "code/external/ext_eval.py", "EXT_AABC_SAMPLE_1119.json": "code/external/aabc_sample.py",
    "EXT_AABC_NULLCHECK_1119.json": "code/external/ext_eval.py (--synthetic, DRY_NULL_SEEDS)",
    "EXT_AABC_DRYRUN_SYNTHETIC_1119.json": "code/external/ext_eval.py (--synthetic)",
}
for f, sc in SCRIPTS.items():
    assert os.path.exists(os.path.join(HERE, "..", sc.split(" ")[0])), sc
json.dump({"levers": LEV, "prereg": PR, "scripts": SCRIPTS,
           "files_read": {f: _sha[f] for f in sorted(_sha)}, "prior_use": PRIOR_USE, "prior_dup": PRIOR_DUP},   # every result file opened, index row or not (checks included)
          open(os.path.join(HERE, "ledger", "tables_meta.json"), "w"), indent=1)
print(f"{len(ROWS)} rows from {len(_cache)} result files")
if "--figs" in sys.argv:
    import figures
    figures.make(HERE, {r["key"]: r["value"] for r in ROWS}, load)
