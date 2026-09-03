# Handoff: Strange MCA → Software Factory

**Purpose.** Context transfer for starting a new Claude Code session on the *software factory* phase. Written 2026-09-02 at the end of a long strange-mca session (Levin-paper walkthrough → instrumentation → experiments). Everything a fresh session needs to pick up the thread is here or linked from here.

**How to use.** Start a new session (from this directory, so project memory also loads) and open with something like: *"Read docs/software-factory-handoff.md, then let's start the software-factory founding doc — I'll lay out my framing first."* If the factory becomes a new repo, copy this file there as its seed context and ask that session to save the key facts to its own memory.

---

## 1. Where things stand

**strange-mca** is a vibe-coded research playground (README disclaimer is deliberate and should carry over as a norm): LLM agents in a tree, each responding to the full task from a perspective, lateral exchange within sibling groups, coordinators observing children, round-based convergence, optional strange-loop self-reflection at the root. Architecture: flat LangGraph `StateGraph`, `ChatOpenAI` agents, gpt-4o-mini by default.

**The pivot.** Jose is moving from the research phase toward a concrete **agent-orchestration system framed as a software factory** — orchestrated agents that build software applications — with topology learnings (below) and stigmergy as founding principles rather than retrofits. Jose's own stated intuition: *topology and stigmergy are very relevant to a software factory* — he wants to articulate why in his own words before the doc is drafted. **Let him lay out his framing first.**

**Decisions made:** two docs, kept separate — (1) `docs/topology-learnings.md` (theory applied to strange-mca; done) and (2) the software-factory founding doc (not started; this handoff feeds it). Instrument-don't-renovate for strange-mca: it stays the experimental apparatus; architectural ideas are filed as issues (#23–#26) for the factory to inherit.

**Decisions open:** (a) **new repo vs. evolve strange-mca** — recommended new repo, strange-mca kept as the research substrate the factory cites; (b) **stack** — current LangGraph + `ChatOpenAI` vs. evaluating the Claude Agent SDK (agent loop, tools, subagents natively). Both are Jose's calls.

## 2. Theory to carry into the factory design

Distilled from Sacco, Sakthivadivel & Levin, *Topological constraints on self-organisation in locally interacting systems* (2026), plus Simon. Full version: `docs/topology-learnings.md`.

- **Topology decides which collective phases are possible; coupling strength (prompts) only shifts constants.** An architecture's wiring can be audited on paper before spending LLM calls. Chains (incl. autoregressive LLMs beyond their context window) cannot hold long-range order at any noise level — coherence has a finite horizon that constants stretch but never remove.
- **Clique hierarchies get a phase flat topologies can't:** dense wiring inside groups makes dissent expensive; sparse wiring between groups keeps divergence cheap → internal unanimity + cross-group diversity, stable in a window with *two* edges (too much coupling = consensus mush/groupthink; too little = disorder). Recursive via supercliques. **Order is a resource you place by choosing where wiring is dense, not a property you have.**
- **Stigmergy is the escape hatch:** when internal wiring can't hold order, offload it into the environment — persistent marks are long-range edges bypassing the chain (termite mounds, notebooks, repos, issues, CI). *For a software factory the repo is the blackboard, arguably for free.*
- **Simon (*Sciences of the Artificial*)** said the same from the design side: near-decomposability (structure: strong-within/weak-between) and the watchmaker parable (process: stable intermediate forms survive interruption). Software engineering is the 60-year confirmation — modules, interfaces, commits as stable intermediates, CI as the interruption test. Factory framing: *wire the agent organization the way we learned to wire the code.*
- **Third failure mode — glassy/jammed:** mixed-sign couplings (adversarial roles) create frustration; runs that neither converge nor oscillate but stick. Expect it in any orchestration with reviewer/critic roles.
- **Fixed vs. dynamic graph:** the theory covers static wiring only (adiabatic assumption); dynamic routing/structure exits it (Watson & Levin, *natural induction*, is the theory for that regime). Noted on strange-mca issue #10.
- **Related, for the factory:** compressed coordination signals beat verbal nudges (*virtual governor*, Lyons/Pio-Lopez/Levin — issue #24); shared scarcity models as cognitive glue (issue #25); learnable novelty / ΦID as emergence metrics (issue #26).

## 3. Empirical state (as of 2026-09-02, end of day)

`docs/experiment-log.md` is authoritative. Instrumentation is merged (PR #27: sibling/cross-group similarity, phase classifier, `lateral_pressure` dial; PR #28: report-side embedding similarity, baseline/Δ metrics, gateway routing, `rescore_report.py`). Eleven runs total across gpt-4o-mini and Haiku 4.5.

**The key result, and a correction.** A 3-task battery on Haiku compared one 6-leaf group against two 2-leaf groups (7 agents each). The paper's clique prediction held 3/3: **small cliques cohered internally** (sibling similarity rose +0.067 over each run's no-interaction baseline) **while diverging from each other** (cross-group similarity fell every round — now 7/7 depth-3 runs ever); **the 6-clique did not cohere** (Δ −0.005). Getting there exposed a framing error in an earlier version of the theory doc, which had predicted the opposite within-group direction by importing strange-mca's design goal (diversity *within* sibling groups) into a theory that puts diversity *between* cliques. Corrected in `topology-learnings.md` §4.

**Design implication, with data behind it:** strange-mca places its perspective diversity inside the densely wired groups — exactly where the topology erodes it. Agents meant to stay distinct belong in *different* cliques; agents meant to agree belong together. This is direct evidence on question 2 below and should shape the factory's org design from the start.

**Other findings to carry:** the lateral-pressure dial moves cross-group similarity monotonically (maintain < balanced < integrate) but does little within groups — it acts through the coordinator clique that couples groups. Dynamics are model-dependent (gpt-4o-mini showed no within-group cohesion; Haiku does): topology sets what is possible, the model sets the effective coupling. Response length roughly triples over three rounds and scales with group size (no scarcity — issue #25). Absolute-threshold phase classification is still uninformative (embedding cosine sits in a model-specific band); the trustworthy readouts are *relative* — Δ from baseline, within- vs cross-group ordering, round-over-round trends. Next instrument step is baseline-relative thresholds.

## 4. The questions the factory doc must answer

Raised in conversation, not yet answered — Jose wants to speak to them in his own framing first:

1. **What does the factory manufacture, and what is its "stored pattern"?** (spec? architecture invariants? house conventions — the thing every part must stay coherent with)
2. **Where should order live, and where should diversity live?** (what must be unanimous — contracts, style, architectural decisions — vs. where uncorrelated exploration is wanted; dense wiring for the former, deliberate sparseness for the latter) — *now with evidence: see §3. Dense groups converge; distinctness survives only across sparse boundaries.*
3. **What is the blackboard?** (the repo, plausibly: code, tests, CI state, issues as durable marks that outlive any agent's context)
4. **What breaks coherence at scale?** (long-horizon agent work is the chain regime: drift, not crash; fix is environmental pinning, not bigger contexts)
5. **What are the factory's stable intermediate forms?** (Simon: units of work small enough to survive interruption and settled enough to build on — likely the single most consequential design choice for agent orchestration)

## 5. Pointers

- Theory: `docs/topology-learnings.md` · Evidence: `docs/experiment-log.md` · Prior design lineage: `docs/design-emergent-mca.md` (and the RFC/design docs it supersedes)
- Issues: #10 (dynamic topology boundary), #23 blackboard, #24 compressed signals, #25 scarcity, #26 metrics · PRs #27 (instrumentation), #28 (embedding metrics + results)
- Reading list beyond the topology paper: Levin & Lyons *Cognitive glues* (2026); Lyons/Pio-Lopez/Levin *Virtual Governor* (2026); Watson/Levin/Lewens *Natural induction* I & II (2025); Pigozzi & Levin ΦID papers; Zhang & Levin *Learnable Novelty* (2026); Simon *The Sciences of the Artificial* ("The Architecture of Complexity", "The Science of Design")
- Infra: local LiteLLM gateway at `http://xochitl:4000` (OpenAI-compatible; local qwen/gpt-oss models and `anthropic/*` incl. Haiku; key via `LITELLM_MASTER_KEY`, see the `local-offload` skill). strange-mca routes chat through it with `MCA_CHAT_BASE_URL` / `MCA_CHAT_API_KEY`. Run long sequential experiments under `caffeinate -i`.

## 6. Working conventions Jose has established

- The repo is the blackboard: design thinking goes into `docs/` (RFC → design doc → implementation is the established lineage), open questions become labeled issues, decisions get recorded where a future session will find them. Claude Code over chat surfaces for anything touching the repo.
- Branch → PR → Jose merges. Adversarial review before pushing (PR #27's 12-agent review caught 9 real defects — worth repeating for non-trivial changes).
- Honesty norms: n=1 is not evidence; say what an instrument can and cannot see; playground disclaimer stays visible.
- Jose dictates many messages (voice → text); read for intent, don't fuss over typos. He prefers being asked before spending API money, and likes options priced out (time + cost) before committing.
