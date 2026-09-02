# Experiment Log

Dated empirical record of what the instrumentation has actually measured. Complements [topology-learnings.md](topology-learnings.md), which is the theory; this file is the state of the evidence. Newest first.

Reminder from the README: this is a playground. Every entry here is a small, informal run — treat findings as observations, not results.

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

## State of the instrument (as of 2026-09-02)

**Trustworthy now:** sibling-group similarity and cross-group similarity as *relative* diversity measures (comparing across agents, where lexical difference is the thing being detected); pre/post-lateral comparisons; response-length trends.

**Not trustworthy yet:** the convergence score and therefore the phase classification's converged/oscillating/stuck distinction, for prose-length responses. Blocked on embedding-based similarity (#26).

**Recommended order of next work:**
1. Embedding-based similarity with round-1 pre-lateral baseline normalization (#26). Small change; unblocks everything below.
2. `lateral_pressure` sweep (`maintain` / `balanced` / `integrate`) on one topology — locates where the coupling dial actually moves the system.
3. Full topology battery (3+ tasks × 2+ topologies). Cheap enough on a small hosted model or free on a local one via an OpenAI-compatible gateway; running the same comparison on a different model also tests the theory's substrate-independence claim.
