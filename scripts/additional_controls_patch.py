# =============================================================================
# PATCH — additional Phase C controls (reviewer objections R2 and R3)
# Apply to the existing notebook. Both cells reuse the cached dataset, the
# existing run log and the resumption logic, so nothing already computed is
# recomputed. Estimated cost on the same A100: ~4.7 GPU-hours total.
# =============================================================================

# ---------------------------------------------------------------- R2 --------
# Phase C at the two remaining initialization scales, removing the
# conditionality of the (P4) conclusion on the calibrated value s = 1.5.
# 2 variants x 4 betas x 8 seeds x 2 scales = 128 runs  (~3.1 GPU-h)

C_INIT_SCALES_EXTRA = [1.0, 2.5]

cells_C2 = [(REF_MODULUS, REF_WIDTH, beta, decay, seed, sc)
            for sc in C_INIT_SCALES_EXTRA
            for beta in C_BETAS for decay in C_DECAYS for seed in C_SEEDS
            if weak_reg_ok(REF_ETA, REF_LAMBDA, beta)]
phase_c2 = run_grid(cells_C2, "C2", "Phase C2 (init-scale robustness)")

# ---------------------------------------------------------------- R3 --------
# Phase C repeated with the grokking component tracked, so that the (P4)
# rate ratio can be formed on the theory's own observable instead of on
# ||theta||, which Phase D shows to be slower by a factor of ~26.
# 2 variants x 4 betas x 8 seeds = 64 runs  (~1.6 GPU-h)
#
# The projector is built once per (beta, seed) at a post-interpolation
# checkpoint of the COUPLED arm and reused for both arms of that pair, so
# that the two trajectories are projected onto the same subspace and the
# ratio remains a like-for-like comparison.

C_TRACK_K0 = 1000          # same convention as Phase D

def run_phase_c_tracked():
    (X, Y, Xt, Yt), _ = DATASETS[REF_MODULUS]
    X_full = torch.cat([X, Xt], dim=0)
    done = {(r["tag"], r["beta"], r["decay_variant"], r["seed"])
            for r in load_records(RUNS_JSONL) if r.get("tag") == "C3"}
    out = []
    for beta in C_BETAS:
        if not weak_reg_ok(REF_ETA, REF_LAMBDA, beta):
            continue
        for seed in C_SEEDS:
            # projector from the coupled arm at C_TRACK_K0
            _, cap = train_run(REF_MODULUS, REF_WIDTH, beta, "coupled_l2", seed,
                               init_scale=REF_INIT_SCALE,
                               capture_state_at=C_TRACK_K0,
                               eval_every=D_EVAL_EVERY, tag="C3_ref")
            if cap is None:
                print(f"  beta={beta} seed={seed}: checkpoint not reached; skipped")
                continue
            set_seed(derive_seed(seed, REF_MODULUS, REF_WIDTH, beta,
                                 "coupled_l2", REF_INIT_SCALE, "C3_ref"))
            model = QuadraticMLP(2 * REF_MODULUS, REF_WIDTH, REF_MODULUS,
                                 init_scale=REF_INIT_SCALE)
            with torch.no_grad():
                model.theta.copy_(cap["theta"])
            U_g, diag = build_grokking_projector(model, X, X_full)
            comp0 = float((U_g.mT @ cap["theta"].to(U_g.dtype)).norm())

            for decay in C_DECAYS:
                if ("C3", beta, decay, seed) in done:
                    continue
                rec, _ = train_run(REF_MODULUS, REF_WIDTH, beta, decay, seed,
                                   init_scale=REF_INIT_SCALE,
                                   track_U=U_g,
                                   component_threshold=D_COMPONENT_FRAC * comp0,
                                   eval_every=D_EVAL_EVERY, tag="C3")
                rec.update({"projector": diag,
                            "component_at_reference": comp0,
                            "projector_source": "coupled_l2 @ %d" % C_TRACK_K0})
                append_record(RUNS_JSONL, rec)
                out.append(rec)
                print(f"  beta={beta:<5} {decay:<13} seed={seed} "
                      f"k_grok={rec['k_grok']} "
                      f"component_rate={rec['component_rate']}")
    return out

phase_c3 = run_phase_c_tracked()

# ------------------------------------------------------------- ANALYSIS -----
# Rate ratio on the component, to be compared with Table 7 of the manuscript.
_c3 = pd.DataFrame([{k: r.get(k) for k in
                     ("beta", "decay_variant", "seed", "k_grok",
                      "component_rate", "component_rate_r2", "grokked")}
                    for r in load_records(RUNS_JSONL) if r.get("tag") == "C3"])
rows = []
for beta in sorted(_c3.beta.dropna().unique()):
    for seed in sorted(_c3.seed.dropna().unique()):
        cp = _c3[(_c3.beta == beta) & (_c3.seed == seed) &
                 (_c3.decay_variant == "coupled_l2")]
        dc = _c3[(_c3.beta == beta) & (_c3.seed == seed) &
                 (_c3.decay_variant == "decoupled_wd")]
        if len(cp) and len(dc) and cp.iloc[0].component_rate and dc.iloc[0].component_rate:
            rows.append({"beta": beta, "seed": seed,
                         "ratio_component_rate":
                             cp.iloc[0].component_rate / dc.iloc[0].component_rate,
                         "predicted": predicted_p4_ratio(beta)})
df_c3 = pd.DataFrame(rows)
if len(df_c3):
    for beta, grp in df_c3.groupby("beta"):
        lo, hi = bootstrap_ci(grp["ratio_component_rate"])
        print(f"beta={beta:<5} predicted={predicted_p4_ratio(beta):6.2f}  "
              f"measured={grp['ratio_component_rate'].mean():7.3f}  "
              f"CI=[{lo:.2f}, {hi:.2f}]  "
              f"contains={'yes' if lo <= predicted_p4_ratio(beta) <= hi else 'no'}")


# ---------------------------------------------------------------- R4 --------
# Raised ceiling at p = 31, resolving the right-censoring reported in the
# manuscript. 72 runs at twice the original ceiling  (~9.6 GPU-h).
#
# Run this ONLY after the two cells above, and note that the records are
# written under a distinct tag so that the censored p=31 results already in
# the log are preserved rather than overwritten.

MAX_STEPS_P31 = 800_000

cells_B31 = [(31, w, REF_BETA, REF_DECAY, s, sc)
             for w in B_WIDTHS for sc in B_INIT_SCALES for s in B_SEEDS]

def run_p31_extended():
    done = {(r["tag"], r["p"], r["width"], r["init_scale"], r["seed"])
            for r in load_records(RUNS_JSONL) if r.get("tag") == "B31"}
    out = []
    for (p, w, beta, decay, seed, sc) in cells_B31:
        if ("B31", p, w, sc, seed) in done:
            continue
        rec, _ = train_run(p, w, beta, decay, seed, init_scale=sc,
                           max_steps=MAX_STEPS_P31, tag="B31")
        append_record(RUNS_JSONL, rec)
        out.append(rec)
        print(f"  p=31 w={w} s={sc} seed={seed} -> "
              f"k_grok={rec['k_grok']} grokked={rec['grokked']} "
              f"final_test_acc={rec['final_test_acc']:.3f} "
              f"({rec['wall_clock_s']:.0f}s)")
    return out

phase_b31 = run_p31_extended()

_b31 = pd.DataFrame([{k: r.get(k) for k in
                      ("width", "init_scale", "seed", "k_mem", "k_grok",
                       "delay_ratio", "grokked", "final_test_acc")}
                     for r in load_records(RUNS_JSONL) if r.get("tag") == "B31"])
if len(_b31):
    print(f"\np=31 at ceiling {MAX_STEPS_P31:,}: "
          f"{int(_b31.grokked.sum())}/{len(_b31)} grokked")
    if _b31.grokked.any():
        print(_b31[_b31.grokked].groupby(["width", "init_scale"])["delay_ratio"]
              .agg(["mean", "std", "count"]).round(2).to_string())
    else:
        print("Still censored: the delay at p=31 exceeds "
              f"{MAX_STEPS_P31:,} iterations, which is itself reportable "
              "as a lower bound.")
