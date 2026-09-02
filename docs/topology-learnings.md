# Topology Learnings: Statistical Mechanics Constraints on Strange MCA

**Author**: Jose Cortez + Claude
**Date**: 2026-09-01
**Status**: Reference notes
**Source**: Sacco, Sakthivadivel & Levin, *Topological constraints on self-organisation in locally interacting systems*, Phil Trans A 384(2320), 2026 ([arXiv:2501.13188](https://arxiv.org/abs/2501.13188), [interactive version](https://francesco215.github.io/Language_CA/), [video abstract](https://youtu.be/cGcY-ReeGDU))

---

## 1. Why this paper matters to strange-mca

Strange-mca's core hypothesis is that globally coherent, emergent behavior can arise from purely local agent interaction. This paper proves that whether *any* locally-interacting system can hold global coherence is decided by its **interaction topology alone** — before dynamics, prompts, or models enter the picture. Coupling strengths, message richness, and window sizes only shift constants; they can never change the verdict.

Practical consequence: **the architecture can be audited on paper, for free, before spending LLM calls.** The graph emitted by `build_agent_tree()` determines which collective phases are possible; prompts and temperature only move the system around inside that possibility space.

---

## 2. The framework, compressed

- **Model**: a graph of units, each holding a state ("spin" = slot with a value). The energy function (Hamiltonian) is a sum of local penalty terms; formally H = H̄ ⊙ G — coupling strengths entry-wise **masked by the adjacency matrix**. Topology is baked into the shape of the loss (this is literally what an attention mask does).
- **Dynamics**: each unit repeatedly re-samples its state via a local softmax over its neighborhood at temperature T. No global controller, no global objective anywhere. Free-energy minimization is bookkeeping about where this random walk statistically ends up, not a mechanism anything computes.
- **The one decisive equation**: ΔF = ΔE − TΔS, evaluated for the move "insert a defect."
  - ΔE = the defect's energy price (how many edges it strains)
  - ΔS = log(number of ways/places the defect can occur)
  - T = noise level (for LLM agents: literally sampling temperature, plus prompt looseness)
  - ΔF < 0 ⇒ the broken family holds exponentially more probability mass (ratio e^(−ΔF/T)); order does not hold
- **Only growth rates matter** (big-O comparison of cost vs. opportunity as system/defect size grows). Universality: strengths, window sizes, and pattern counts wash out (Lemmas 1–2, Thm 1). The self-organisation verdict of *any* local system on graph G equals that of nearest-neighbor Ising on G (Cor 1) — so any candidate topology can be checked against a century of known Ising results.
- **The defect that matters is the domain wall**: a seam between two *locally valid* regions (a merge conflict between internally-consistent branches). The interior of a "wrong" region is locally perfect and hence invisible to every local check (Markov blanket screening); only the seam is under tension. Order lives or dies on seam economics.

## 3. Verdicts by topology

| Topology | Defect price | Defect opportunities | Verdict |
|---|---|---|---|
| **Chain (1D)** | Flat fee (seam = a point) | Grows with length | No ordered phase at any T > 0. Coherence horizon L* ~ e^(ε/T): stretchable by constants, never removable |
| **Grid (2D+)** | Grows with defect boundary | Grows comparably | Ordered phase below a critical temperature |
| **Clique hierarchy** | Steep *inside* groups, thin *between* groups | — | **Hierarchical phase**: internal unanimity + cross-group diversity, stable in a temperature window (Thm 4, Prop 3); recursive via supercliques |

- **LLMs are chains**: autoregressive generation is Boltzmann sampling under the Hamiltonian H = −log P(sequence) (surprisal = energy; Thm 3); causally-masked attention is a modern Hopfield network, i.e., an energy-based spin collective (Prop 2). Within-context, attention is a complete graph (order is fine); the chain pathology applies at scales beyond the window. Context growth is the pushing of a constant.
- **The clique-hierarchy moral**: *order is a resource you place, not a property you have.* Dense wiring buys unanimity where you want it; sparse wiring preserves diversity where you want that. The chain's "doom" is a tool when used at the scales where variation is desired.

---

## 4. The strange-mca mapping

The paper's Section VI analyzes, near enough, strange-mca's exact graph.

| Paper object | strange-mca object |
|---|---|
| Clique (dense group) | Sibling group with full lateral visibility (`leaf_lateral`, coordinator laterals) |
| Effective graph of cliques | Coordinator level; the tree is a clique hierarchy by construction |
| Markov blanket | Visibility rules table (design doc §7.2) |
| Coupling J | Lateral-prompt pressure (integrate ↔ "MAINTAIN your perspective") |
| Temperature T | Effective noise of generation: prompt looseness + inherent sampling stochasticity. (Sampling temperature is a minor, provider-tuned component — modern post-trained models respond to it with lexical jitter more than substantive variation. Since T is effectively fixed, the controllable surface for the phase-deciding ratio J/T is J: lateral-prompt pressure.) |
| Clique size n_i | `child_per_parent` |
| Number of cliques ℓ | Number of sibling groups |
| Ordered/stored pattern | A globally settled collective answer |
| Domain wall | Divergence between internally-coherent subtrees |

**Target behavior = the hierarchical phase.** Each sibling group internally coherent, groups genuinely differing, root reading the mosaic rather than melting it. This is precisely the design doc's stated goal (perspective diversity + emergent synthesis), now identified with a thermodynamically *stable* phase rather than a hoped-for transient.

**The design doc's failure modes are the flanking phases:**

- *Consensus mush* (Q3 red flag: sibling similarity > 0.8) = fully **ordered** phase — coupling too strong / T too low; the mosaic melts into global unanimity.
- *Oscillation / non-convergence* = **disordered** phase — T too high for even intra-group coherence.
- **Third mode, previously undocumented: glassy/jammed.** Mixed-sign couplings (e.g., a "critical" perspective is a *negative* coupling by design) create frustration → rugged landscape → runs that neither converge nor oscillate but stick in history-dependent disagreement, insensitive to more rounds. Signature: flat, sub-threshold `convergence_scores` trajectory. (The paper's authors flag glassy phases as their forthcoming work.)

**Testable prediction (cheap) — corrected 2026-09-02.** Prop 3's window condition favors **many small groups over few large ones** (the window narrows sharply with clique size). Stated the way the *paper* states it: small cliques should reach **internal cohesion** (siblings converging toward each other) while **diverging from other cliques**; a large clique should fail to cohere. An earlier version of this paragraph inverted the within-group half — it predicted lower sibling similarity for small groups, importing strange-mca's design goal (diversity *within* sibling groups) into a theory that puts diversity *between* cliques. The battery in `experiment-log.md` (Haiku, 3 tasks) supported the paper's version 3/3: two-leaf groups converged (Δ +0.067 over baseline), the six-leaf group did not (Δ −0.005), and cross-group similarity fell every round.

**Design implication (with data behind it):** strange-mca currently places its perspective diversity *inside* the densely wired sibling groups — precisely where the topology erodes it. If perspectives are meant to stay distinct, they belong in *different* cliques; agents meant to agree belong together. The design doc's Q3 red flag ("sibling similarity > 0.8 = diversity collapsing") is therefore in tension with the physics: within-clique convergence is the ordered phase working, not failing — the question is only whether the *right* agents were grouped.

**Observability implication:** root-output Jaccard measures wording stability only. A phase-aware report would also track a per-group order parameter (sibling similarity trajectory = intra-clique magnetisation) and cross-group divergence — i.e., *where* order is forming, not just whether the root's text settled.

---

## 5. Simon: the same architecture, discovered from the design side

Herbert Simon, *The Sciences of the Artificial* — specifically "The Architecture of Complexity" (1962 essay, later the book's spine) — made Section VI's claim sixty years earlier, in words, from the design/engineering side rather than the physics side. (Provenance note: the correspondences below are our synthesis, not claims made by either source.)

Simon's argument has two distinct parts, and both matter here — one is about **structure**, one about **process**:

**Near-decomposability (structure).** Complex systems that persist are "nearly decomposable": strong/fast interactions *within* subsystems, weak/slow interactions *between* them. This is, edge-for-edge, the clique hierarchy — dense intra-group wiring, sparse bridges. Simon offered it as an empirical generalization; the topology paper derives what it buys (the hierarchical phase, stable within a temperature window). Simon's observation, issued a phase diagram.

**The watchmaker parable (process).** Hora builds watches from stable ~10-part subassemblies; Tempus builds monolithically and loses everything on each interruption. Simon's arithmetic makes Tempus thousands of times slower. The form of the argument is a free-energy race: Tempus needs 1000 uninterrupted steps, ~e^(−1000p); Hora only ever needs 10, ~e^(−10p) per module — interruption rate playing temperature, assembly size playing the scale over which order must be held. "Stable intermediate forms" are free-energy basins at intermediate scale: order locked in at module size, so noise destroys at most one module's worth. Hora's next level assembles settled subassemblies as single units — the superclique recursion, as kinetics.

| Simon | Topology paper |
|---|---|
| Near-decomposability (strong within, weak between) | Clique hierarchy (dense intra-, sparse inter-clique) |
| Stable intermediate forms | Free-energy basins at intermediate scales; clique-level ordered phases |
| Watchmaker assembly under interruption, (1−p)^n | Order-vs-noise race, e^(−ΔF/T); recursion via effective spins |
| Empty world hypothesis (most interactions absent) | Sparse adjacency matrix G — structure *is* the pattern of missing edges |
| Bounded rationality (each level needs only local knowledge) | Markov blanket (each vertex computes only over its neighbors) |

**What the paper adds to Simon: necessity and the window's second edge.** Simon argued hierarchy is advantageous; he never proved flat topologies *can't* hold order (the no-go theorems do), and his framework has no failure mode for *over*-integration. The physics does: couple too strongly / cool too far and the hierarchy melts into a monolith of consensus — for strange-mca, the mush failure; for an organization, groupthink. "Hierarchies are stable" is true specifically *inside* the temperature window, which has two edges.

**What Simon adds to the paper: kinetics, function, and the design stance.** The paper is equilibrium-only — what phases can exist. Simon is about assembly — how complexity is reached under noise (exactly what the paper's §VII concedes it doesn't cover). They compose: Simon says hierarchy is how you *get there*; the paper says hierarchy is how you *stay there*. And Simon's subsystems do different *jobs* connected by interfaces — functional decomposition, not just an agreement game.

**Caveat on the mapping's edge semantics.** An Ising coupling is symmetric peer pressure toward agreement; a software interface is a directed, constrained channel. The sparse/dense skeleton transfers cleanly; the semantics of individual edges does not, fully. Treat the correspondence as structural, not literal.

**Bridge to the software factory (forthcoming doc).** Software engineering is a sixty-year experimental confirmation of the watchmaker parable: modules, interfaces, commits as stable intermediates, CI as the interruption test — Hora's strategy, rediscovered by an industry that kept getting burned playing Tempus. The factory question, in Simon's terms: wire the *agent organization* the way we learned to wire the code — near-decomposable, stable intermediates at every scale, order placed at the boundaries.

---

## 6. Boundaries of applicability

1. **Fixed-graph assumption.** The theory assumes edges never change (adiabatic approximation). strange-mca satisfies this *exactly* (`build_agent_tree()` freezes the graph), so the results apply cleanly — and dynamic routing/structure ([issue #10](https://github.com/josecodes/strange-mca/issues/10), see [the comment](https://github.com/josecodes/strange-mca/issues/10#issuecomment-5397534707)) exits the covered regime. The relevant theory for state-dynamics-rewiring-topology is Watson, Levin & Lewens, *Evolution by natural induction* I & II (Interface Focus 15(6), 2025).
2. **Equilibrium only.** The theorems are equilibrium statistical mechanics (Landau theory). Three rounds of prompted revision is not equilibrium. Read the verdicts as *directional pulls* — which regime the wiring and knobs push toward — not as guarantees about round 3.
3. **Dynamics are approximate.** Prompted revision is only loosely a local softmax sampler; the mapping of J and T to prompts and sampling settings is qualitative.

---

## 7. Stigmergy

The paper's closing argument: since internal wiring often *cannot* hold long-range order (the no-go), evolution favors offloading order into the environment — stigmergy. A persistent environmental mark is a **long-range edge that bypasses the chain** (equivalently: the environment is a durable, promiscuously-connected hub vertex). Termite mounds, ant trails, notebooks; for LLM systems: retrieval, scratchpads, repos, issues.

strange-mca currently coordinates by pure direct message-passing; nothing persists as a medium. A **blackboard / shared workspace** that agents read and write across rounds would add the environment-hub to the clique hierarchy — coherence storage independent of any agent's window, and the most theory-endorsed upgrade available for convergence-across-rounds. This thread continues in the software-factory design doc (forthcoming), where stigmergy is a founding principle rather than a retrofit.

---

## 8. Related follow-on reading

- **Watson, Levin & Lewens** — *Evolution by natural induction* I & II (Interface Focus 15(6), 2025): adaptation via connections that "give way under stress" — the dynamic-topology regime beyond this paper's fixed-graph boundary; the theory to reach for if issue #10 proceeds.
- **Lyons, Pio-Lopez & Levin** — *Alignment Is to a Virtual Governor* ([preprint, 2026](https://www.preprints.org/manuscript/202607.0220)): coordination via compressed signals translating system-level stress into component-level stress; parts never represent the global goal. Directly challenges the verbal-nudge design of `create_signal_prompt` (compare: compressed scalar signals vs. 2–3 sentence prose).
- **Levin & Lyons** — *Cognitive glues are shared models of relative scarcities* (Phil Trans A 384(2320), 2026, [doi:10.1098/rsta.2024.0528](https://doi.org/10.1098/rsta.2024.0528)): what binds a collective is a shared scarcity model (canonical example: prices). strange-mca has no scarcity — nothing makes one agent's contribution cost another's — which may be why synthesis tends toward averaging.
- **Pigozzi, Goldstein & Levin** — *Associative conditioning... increases integrative causal emergence* (Communications Biology 8:1027, 2025) and **Pigozzi & Levin** ([arXiv:2605.06746](https://arxiv.org/abs/2605.06746)): ΦID / causal emergence as a *computable* answer to evaluation question Q2 ("does the root produce something no child did") — needs per-agent per-round embeddings; `agent_history` already has the right shape.
- **Zhang & Levin** — *Intelligence from Learnable Novelty* ([arXiv:2607.18433](https://arxiv.org/abs/2607.18433)): cheap differentiable metric separating learnable novelty from noise; candidate upgrade for the Jaccard convergence check (loop while rounds add learnable structure; stop when residual novelty is noise). Caveat: built for long time series; max_rounds=3 may be too short to estimate.

---

## 9. Empirical status

See [experiment-log.md](experiment-log.md) for what has actually been measured. As of 2026-09-02 (evening): 11 runs across gpt-4o-mini and Haiku 4.5, Jaccard and embedding metrics. The clique-hierarchy prediction, stated the paper's way, is supported 3/3 tasks on Haiku (small cliques cohere, the large one doesn't, groups diverge from each other); the cross-group-divergence "mosaic" signature holds in 7/7 depth-3 runs; the lateral-pressure dial acts monotonically on cross-group similarity but not within groups; dynamics are model-dependent (gpt-4o-mini showed no within-group cohesion). Absolute-threshold phase classification remains uninformative — relative metrics are the signal.
