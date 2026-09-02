# Experiment Log

Dated empirical record of what the instrumentation has actually measured. Complements [topology-learnings.md](topology-learnings.md), which is the theory; this file is the state of the evidence. Newest first.

Reminder from the README: this is a playground. Every entry here is a small, informal run — treat findings as observations, not results.

---

## 2026-09-02 (evening) — Haiku pressure sweep + topology battery, embedding metrics

**Setup.** Claude Haiku 4.5 via the local LiteLLM gateway (`MCA_CHAT_BASE_URL`), `max_rounds=3`, downward signals on, `similarity_method=embedding` (OpenAI `text-embedding-3-small`; provisional thresholds mush 0.97 / stability 0.95 / convergence 0.95). The convergence loop itself still runs on Jaccard, so every run went the full 3 rounds. ~45 LLM calls per run, 9 runs.

- **Sweep:** cpp=2/depth=3 × lateral pressure {maintain, balanced, integrate} × the transit task — 3 runs.
- **Battery:** {cpp=6/depth=2 (one 6-leaf group), cpp=2/depth=3 (two 2-leaf groups)} × balanced × 3 tasks (transit, food waste, platform liability) — 6 runs, 7 agents per run in both arms.

**Battery — one big group vs two small groups (balanced pressure, mean of 3 tasks):**

| | cpp=6 / depth=2 | cpp=2 / depth=3 |
|---|---|---|
| Final leaf sibling similarity | 0.841 (sd 0.017) | 0.868 (sd 0.021) |
| Baseline (round-1, pre-lateral) | 0.847 | 0.800 |
| **Δ final − baseline** | **−0.005** (−0.004, +0.043, −0.055) | **+0.067** (+0.085, +0.101, +0.016) |
| Paired Δ difference by task (small − big) | | +0.089, +0.058, +0.071 |
| Final cross-group similarity | — | 0.777 (fell every round in all 3 runs) |
| Non-root agent stability | 0.852 | 0.853 |
| Leaf response length, rounds 1→3 (words) | 748 → 2378 | 491 → 1445 |
| Genuinely converged / phase | 0/3, all "oscillating" | 0/3, all "oscillating" |

**Sweep — pressure dial on cpp=2/depth=3 (transit task):**

| Pressure | Sibling sim (final) | Baseline | Δ | Cross-group (final) | Cross-group by round |
|---|---|---|---|---|---|
| maintain | 0.857 | 0.797 | +0.060 | 0.768 | 0.821 → 0.797 → 0.768 |
| balanced | 0.902 | 0.796 | +0.106 | 0.788 | 0.828 → 0.781 → 0.788 |
| integrate | 0.893 | 0.822 | +0.071 | 0.811 | 0.849 → 0.843 → 0.811 |

**Findings.**

1. **The prediction as previously written in `topology-learnings.md` §4 was stated the wrong way round — and the paper's actual prediction was supported, 3 tasks out of 3.** The doc had said "smaller groups should preserve perspective diversity better (lower sibling similarity)." What happened: in the two-small-groups arm, siblings *converged toward each other* over the rounds (Δ +0.067) while in the one-big-group arm they did not (Δ −0.005) — paired difference positive in every task. That is the hierarchical phase exactly as the paper defines it: **within-clique cohesion plus cross-group divergence**, appearing in the small-clique topology and failing to appear in the 6-clique — Prop 3's claim that large cliques have a narrow-or-absent window. The earlier framing had imported strange-mca's design goal (diversity *within* sibling groups) into a theory that puts diversity *between* cliques. Design implication, now with data behind it: strange-mca places its perspective diversity inside the densely wired groups, i.e. exactly where the topology erodes it. If perspectives are meant to stay distinct, they belong in different cliques; agents that are meant to agree belong together. (Corrected in `topology-learnings.md` §4.)
2. **The mosaic signature is the most robust finding in the dataset.** Cross-group similarity fell round over round in all six depth-3 runs here; combined with the earlier gpt-4o-mini run, that is 7 of 7 depth-3 runs ever executed, across two models and two similarity metrics.
3. **The lateral-pressure dial acts on the cross-group axis, not within groups.** Within-group: balanced pulled hardest (+0.106), integrate less (+0.071), maintain least (+0.060) — non-monotonic and only ~2× the noise floor. Cross-group: maintain 0.768 < balanced 0.788 < integrate 0.811 — monotonic in the predicted direction. Structurally sensible: what couples the two groups is the coordinator clique's lateral exchange, so pressure on coordinator laterals is what moves groups toward or away from each other.
4. **Dynamics are model-dependent.** On gpt-4o-mini (the earlier run, re-scored under embeddings) three rounds left siblings at baseline in both topologies (Δ −0.016 / −0.028); on Haiku the small-group cohesion appears. Consistent with the theory's division of labor — topology sets which phases are possible, the model sets the effective coupling — but the 4o-mini comparison is n=1 per arm.
5. **Response bloat is severe on Haiku and scales with group size.** Leaves tripled in length over three rounds; the 6-leaf group reached ~2400 words per leaf vs ~1450 for 2-leaf groups. More peers to absorb, more text (issue #25).
6. **The phase classifier is still uninformative, for a new reason.** All 9 runs are "oscillating" because absolute embedding thresholds calibrated on gpt-4o-mini re-scores do not transfer: Haiku's whole similarity band sits lower (baseline ~0.80 vs 0.93; root round-over-round 0.87–0.94, under the 0.95 convergence threshold). The next step is baseline-relative thresholds (e.g. collapse = final exceeds the run's own baseline by a margin), not another absolute calibration.
7. **Noise floor.** Baselines are pressure-independent by construction and varied ±0.015 across the sweep's three runs; across tasks within a config they varied 0.76–0.89. Differences under ~0.03 in a single comparison should be ignored; the paired-by-task Δ comparison (finding 1) is the cleanest test in this dataset.

**Caveats.** n=3 per arm (battery), n=1 per setting (sweep); one model per experiment; absolute sibling similarity is confounded by which perspectives share a group (Δ from baseline is the within-run control); embedding thresholds provisional; Haiku through the gateway ran ~20 min per run, not the ~5 min estimated.

Reproduce (key in `LITELLM_MASTER_KEY`; see the local-offload notes):
```bash
export MCA_CHAT_BASE_URL=http://xochitl:4000 MCA_CHAT_API_KEY=$LITELLM_MASTER_KEY
caffeinate -i poetry run python -u scripts/topology_experiment.py --configs 2,3 \
  --lateral_pressures maintain balanced integrate --tasks "<transit task>" \
  --model anthropic/claude-haiku-4-5 --similarity_method embedding --output_dir output/haiku_pressure_sweep
caffeinate -i poetry run python -u scripts/topology_experiment.py --configs 6,2 2,3 \
  --model anthropic/claude-haiku-4-5 --similarity_method embedding --output_dir output/haiku_topology_battery
```

---

## 2026-09-02 — Small topology comparison (1 task, 2 topologies)

**Setup.** `scripts/topology_experiment.py`, gpt-4o-mini, `max_rounds=3`, `lateral_pressure=balanced`, downward signals on. Task: "What are the most important trade-offs in designing a public transit system for a mid-sized city?" One run per topology, 7 agents each.

| | cpp=6 / depth=2 (one 6-leaf group) | cpp=2 / depth=3 (two 2-leaf groups) |
|---|---|---|
| Final leaf sibling similarity | 0.259 | 0.246 |
| Sibling similarity by round | 0.241 → 0.270 → 0.259 | 0.226 → 0.230 → 0.246 |
| Cross-group similarity by round | — | 0.227 → 0.207 → 0.189 |
| Root convergence scores | 0.396 → 0.471 | 0.363 → 0.348 |
| Non-root agent stability | 0.278 | 0.266 |
| Lateral revision rate | 0.857 | 0.857 |
| Mean leaf response length (words) | 580 → 719 → 686 | 537 → 581 → 588 |
| Phase classification | oscillating | oscillating |
| LLM calls | 42 | 48 |

**Findings.**

1. **Headline prediction (many small groups preserve diversity better): direction consistent, magnitude meaningless.** Smaller groups scored lower sibling similarity (0.246 vs 0.259), but a 0.013 gap on n=1 is noise. Not evidence either way.
2. **Mosaic signature present in the depth-3 run.** Cross-group similarity (0.189) was below within-group similarity (0.246), and cross-group similarity *fell every round* while within-group drifted up — groups cohering internally while diverging from each other, which is the hierarchical-phase signature from the theory (§3–4 of the learnings doc). Caveats: one run, small magnitudes, confounded by which perspectives were grouped together (analytical+creative vs critical+practical).
3. **Lateral communication under `balanced` pressure produces churn without convergence.** Pre- vs post-lateral sibling similarity per round: 0.251→0.241, 0.285→0.270, 0.263→0.259 (big group); flat/mixed in the small groups. Agents rewrote ~73% of their tokens each round (stability ~0.27) yet final sibling similarity sits at the round-1 *pre-interaction* baseline. The MAINTAIN instruction holds completely; `balanced` sits firmly on the diversity side of the coupling dial. The `integrate` setting is untested and is the obvious next sweep.
4. **The convergence metric is blind to paraphrase.** Neither run converged; both classified "oscillating." Jaccard on 500–700-word prose treats rephrasing as instability, so the 0.85 threshold is effectively unreachable and the phase classifier can only emit "oscillating" for prose-length responses. The big-group root's score *rose* (0.40 → 0.47), a settling the metric half-sees. Issue #26 (embedding similarity) is a prerequisite for the classifier to say anything true about convergence, not an upgrade.
5. **Response bloat scales with group size.** Leaves grew every round, more so in the 6-leaf group (more peers to absorb → more text). Consistent with the no-scarcity diagnosis in issue #25.
6. **Jaccard's floor is task-dependent.** Round-1 pre-lateral sibling similarity was ~0.25/0.23 here versus ~0.16 on the July photosynthesis run. Each run's round-1 pre-lateral value is the natural baseline and should be reported as such (not yet implemented).

**Operational note.** The first attempt hung for ~15 hours after the laptop slept overnight (stale TCP connections to the API, 1.5 s of CPU consumed). Run long sequential experiments under `caffeinate -i` on macOS.

Reproduce:
```bash
caffeinate -i poetry run python -u scripts/topology_experiment.py \
  --tasks "What are the most important trade-offs in designing a public transit system for a mid-sized city?" \
  --max_rounds 3 --output_dir output/topology_experiment_small
```

---

## 2026-07-22 — Retro-scored run (phase metrics applied after the fact)

**Setup.** "Explain the concept of photosynthesis", gpt-4o-mini, cpp=2/depth=2 (analytical + creative leaves), `max_rounds=2`, signals on. Scored with `build_mca_report` from PR #27 over the saved `final_state.json`.

| Metric | Value |
|---|---|
| Sibling similarity by round | 0.157 → 0.175 |
| Root convergence score | 0.321 |
| Non-root agent stability | 0.212 |
| Phase classification | oscillating |
| Leaf response length (L2N1, words) | 343 → 418 → 453 |
| Downward signal length (words) | 134, then 94 (prompt asks for 2–3 sentences) |

**Findings.** Diversity fully preserved (no mush). Root syntheses were thematically near-identical across rounds yet scored 0.32 — first observation of the paraphrase blindness in finding 4 above. Signals overproduce relative to the "brief" instruction, the verbal-nudge pattern issue #24 questions.

---

## State of the instrument (as of 2026-09-02, evening)

**Trustworthy now:** the *relative* metrics — Δ from the run's own round-1 pre-lateral baseline, within- vs cross-group ordering, round-over-round trends — under either similarity method. Embedding similarity (`--similarity_method embedding`, report-side) sees through paraphrase and is the default for analysis going forward; Jaccard remains the loop's convergence check.

**Not trustworthy yet:** absolute-threshold phase classification. Jaccard calls prose-length runs "oscillating"; embedding cosine is compressed into a model-specific band, so thresholds calibrated on one model mislabel another. Every run to date carries the label "oscillating" for instrument reasons, not behavioral ones.

**Recommended next work:**
1. Baseline-relative phase thresholds (collapse / convergence / stability expressed as margins over the run's own baseline band), replacing the absolute `PHASE_THRESHOLDS["embedding"]`.
2. Test the design implication of finding 1 above: a topology variant that places distinct perspectives in *different* sibling groups (agents meant to agree grouped together) and checks whether cross-group divergence carries the diversity.
3. Repeat the battery on gpt-4o-mini and a local model (free via the gateway) to separate topology effects from model effects — the theory's substrate-independence claim.
4. Response-length control (a scarcity mechanism, issue #25) — bloat is now the most visible pathology in the data.
