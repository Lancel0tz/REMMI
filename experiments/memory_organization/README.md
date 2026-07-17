# Memory-organization experiments (the *Episodic* pillar of REMMI)

Three strategies for organizing long-term personal memory before/while
answering, evaluated on the ATM-Bench agent harness (`agent_systems/`).

> **Terminology (2026-07-14, repo-consistent):** **`org_dynamic`** now refers to
> REMMI's canonical **inject pipeline** — retrieve with the frozen
> `config/best_routing_config*.json` (routing + reranker), inject the top-k
> evidence into the answerer prompt, single LLM call (the pipeline behind the
> repo's headline numbers). The agentic self-organisation variant that builds a
> per-question `timeline.md` inside an agent loop — previously called
> org_dynamic in this document — is renamed **`org_timeline`** and is
> **deprecated** as a method direction; its results below are kept for the
> capability-interaction analysis. org_dynamic (inject) results with
> gpt-5-mini / gpt-5.5 answerers are being added.


> **Provenance.** This is a local **rebuild** of exploration originally run on a
> cluster whose disk is currently unrecoverable. The code here was re-implemented
> from the author's notes/recollection — it is *not* the original scripts, and
> the qualitative findings below are **recalled results, not reproduced numbers**.
> Original runs used **Qwen3.6-27B** on the **Pi** harness; the local rebuild
> defaults to **GPT-5.5 on Codex** (no local GPU) — both harnesses are wired.

## The three strategies

| Mode | What happens | Sandbox memory files |
|------|--------------|----------------------|
| `org_timeline` | **Dynamic (per-question) organisation.** The answering agent explores only the memory relevant to the current question, organises it into a scratch `timeline.md`, then answers from that view. Prompt-only variant. | SGM (unchanged) |
| `org_static` | **Static (full upfront) agent organisation.** An LLM organiser reworks the *whole* corpus into titled/summarised events offline (`remmi/organize/agent_static.py`, map-reduce over an OpenAI-compatible endpoint); the answering agent reads that index. | SGM + LLM-built `organized_memory.json` |
| `org_heuristic` | **Pure heuristic organisation.** Deterministic day-gap + city-change clustering into events/trips (`remmi/organize/heuristic.py`); no LLM. On the real corpus: 4,292 items → ~264 events (~131 trips), index ≈ 0.8 MB vs 29 MB full SGM. | SGM + heuristic `organized_memory.json` |

All organized modes keep the full per-item SGM files in the sandbox — recall
questions must still answer with exact memory item ids.

## Recalled findings from the original (unrecovered) server runs

Setting: Qwen3.6-27B answerer, Pi harness, ATM-Bench-Hard.

- **Dynamic** — the winner: **saved tokens AND improved QS.** Organising only
  question-relevant memory keeps context small while giving the answerer a
  coherent event view for multi-evidence aggregation.
- **Static** — full upfront agent organisation **neither saved tokens nor
  improved QS** meaningfully: the agent still had to verify against per-item
  records, and the upfront pass burns its own tokens.
- **Heuristic** — **saved tokens but did not improve QS**: the compact index
  narrows the search cheaply, yet event boundaries alone don't help the hard
  multi-evidence aggregation step.

Treat these as hypotheses to re-verify with the scripts below, not as numbers.

## Held-out generalization (out-of-distribution overfit check)

Because the hard split is only 31 questions in total, no in-distribution
held-out is possible; we test cross-distribution transfer on a 59-question
stratified sample of the STANDARD split (single-hop, seed 42, zero overlap with
hard-31). This over-tests generalization (questions AND difficulty both change),
so a retained advantage is strong evidence, a diminished one is confounded with
the difficulty shift.

| gpt-5-mini, held-out | QS | number | recall | open_end |
|----------------------|---:|---:|---:|---:|
| baseline | 55.4 | 52.4 | 46.4 | 60.0 |
| remmi (v9) | 55.4 | 47.6 | 46.4 | 63.3 |
| remmi_weak / scaffold (v8) | 55.4 | 42.9 | 46.4 | 66.7 |

Net QS is identical (55.4) for all three — but this is a *cancellation*, not a
null effect, and the per-qtype breakdown de-confounds it:

- **open_end rises monotonically** (60.0 → 63.3 → 66.7): the multi-evidence
  organisation/scaffold benefit **transfers out-of-distribution**, so the
  scaffold is NOT merely fit to the 31 hard questions.
- **number falls monotonically** (52.4 → 47.6 → 42.9): on easy single-hop
  number questions (baseline already 52.4) the tool workflow adds overhead
  without benefit; the *mandated* scaffold (42.9) hurts more than the *optional*
  tool (47.6) — consistent with the mandate-is-harmful finding.

**Verdict:** the scaffold's mechanism generalizes (not overfitting), but its
large hard-split gain (+17.9) is regime-specific — realized only where the base
task is hard and the baseline is weak; on easy questions the number regression
cancels the open-ended gain. Report v8's advantage as difficulty-conditioned,
not universal. (5.5 held-out for remmi/v9 pending quota windows.)

## Master results table (all methods × both answerers, USD costs)

ATM-Bench-Hard, 31 questions, Codex harness, ATM judge `gpt-5-mini`.
**Cost/Q** = mean USD per question, computed from the per-turn token splits
(uncached input × input rate + cached input × cache-read rate + output ×
output rate) with rates from the local **Tokdash pricing DB** (falls back to a
frozen snapshot; regenerate any time with
`experiments/memory_organization/compare_modes.py`). **$/QS-pt** = total USD ÷
QS — dollars per quality point, the headline cost-effectiveness metric.
Judge cost (~$0.01/leg) excluded. Per-model champion marked 🏆.

#### gpt-5-mini  <sub>(rates: in $0.25/M · out $2/M · cache-read $0.025/M — Tokdash pricing DB)</sub>

| Method | QS | num / rec / open | Billed/Q | Uncached/Q | **Cost/Q** | **$/QS-pt** | Total $ | Unk% | Note |
|--------|---:|:--:|--------:|----------:|-------:|--------:|-------:|---:|------|
| **sgm (baseline)** | **20.4** | 17 / 36 / 8 | 301k | 65k | $0.028 | $0.042 | $0.86 | 29 | no organisation |
| **org_heuristic** | **15.3** | 17 / 31 / 0 | 370k | 83k | $0.034 | $0.069 | $1.06 | 19 | day-gap index, no LLM |
| **org_static** | **24.7** | 17 / 39 / 15 | 451k | 111k | $0.043 | $0.054 | $1.34 | 19 | pure-LLM index, offline (~328k once) |
| **org_hybrid** | **27.8** | 17 / 38 / 23 | 515k | 103k | $0.046 | $0.051 | $1.42 | 16 | index-guided dynamic |
| **org_timeline** | **22.3** | 17 / 41 / 8 | 551k | 120k | $0.051 | $0.072 | $1.59 | 29 | self-organise, no tool |
| **org_remmi (v1)** | **33.2** | 17 / 44 / 31 | 542k | 128k | $0.049 | $0.046 | $1.53 | 6 | retrieval tool, mandated |
| **remmi_weak (v8)** 🏆 | **38.3** | 33 / 41 / 38 | 425k | 95k | $0.039 | $0.032 | $1.22 | 13 | tool mandated + typed + verified |
| **remmi (v9)** | **25.2** | 17 / 40 / 15 | 425k | 100k | $0.041 | $0.051 | $1.29 | 19 | tool OPTIONAL + self-organise |

<sub>org_remmi ladder (mini, superseded iterations):</sub>

| Method | QS | Billed/Q | Uncached/Q | Cost/Q | Note |
|--------|---:|--------:|----------:|-------:|------|
| org_remmi2 | 18.7 | 215k | 58k | $0.024 | rich show + cmd budget |
| org_remmi3 | 12.6 | 161k | 35k | $0.017 | slim show + hard budget |
| org_remmi4 | 21.7 | 420k | 101k | $0.040 | no budget + thoroughness |
| org_remmi5 | 25.6 | 358k | 87k | $0.035 | + answer typing A/B |
| org_remmi6 | 35.5 | 402k | 87k | $0.036 | + 3-way typing |
| org_remmi7 | 21.5 | 399k | 106k | $0.041 | + event-layer tool (neg) |

#### gpt-5.5  <sub>(rates: in $5/M · out $30/M · cache-read $0.5/M — Tokdash pricing DB)</sub>

| Method | QS | num / rec / open | Billed/Q | Uncached/Q | **Cost/Q** | **$/QS-pt** | Total $ | Unk% | Note |
|--------|---:|:--:|--------:|----------:|-------:|--------:|-------:|---:|------|
| **sgm (baseline)** | **36.3** | 33 / 69 / 8 | 296k | 87k | $0.618 | $0.527 | $19.16 | 0 | no organisation |
| **org_heuristic** | **35.4** | 17 / 75 / 8 | 254k | 84k | $0.603 | $0.528 | $18.71 | 0 | day-gap index, no LLM |
| **org_static** | **32.2** | 17 / 75 / 0 | 266k | 71k | $0.553 | $0.533 | $17.15 | 0 | pure-LLM index, offline (~328k once) |
| **org_hybrid** | **32.9** | 17 / 77 / 0 | 419k | 97k | $0.787 | $0.743 | $24.41 | 0 | index-guided dynamic |
| **org_timeline** | **49.8** | 33 / 62 / 46 | 434k | 93k | $0.770 | $0.479 | $23.87 | 0 | self-organise, no tool |
| **org_remmi (v1)** | **33.2** | 33 / 69 / 0 | 281k | 67k | $0.559 | $0.522 | $17.32 | 0 | retrieval tool, mandated |
| **remmi_weak (v8)** | **32.8** | 33 / 68 / 0 | 379k | 85k | $0.670 | $0.633 | $20.77 | 0 | tool mandated + typed + verified |
| **remmi (v9)** 🏆 | **56.1** | 33 / 78 / 46 | 377k | 89k | $0.714 | $0.395 | $22.15 | 0 | tool OPTIONAL + self-organise |

## Local rebuild results (2026-07-03, final)

Setting: **gpt-5-mini** (medium reasoning) on the **Codex** harness, ATM-Bench-Hard
(31 questions), ATM judge `gpt-5-mini`, single run per mode, API billing.
*Billed* tokens count cached context re-sends of the agent loop; *Uncached/Q*
is real new content (new input + output) — the comparable measure.

| Mode | QS ↑ | Billed | Mean/Q | **Uncached/Q** | Unknown% | Organiser cost |
|------|-----:|-------:|-------:|---------------:|---------:|---------------:|
| sgm (baseline) | 20.4 | 9.34M | 301k | 65k | 29% | — |
| org_heuristic | 15.3 | 11.47M | 370k | 83k | 19% | 0 (no LLM) |
| org_static (pure) | 24.7 | 13.99M | 451k | 111k | 19% | 327.6k once (≈10.6k/Q) |
| org_timeline | 22.3 | 17.08M | 551k | 120k | 29% | in-session, ~99% of tokens |
| org_hybrid | 27.8 | 15.97M | 515k | 103k | 16% | reuses static index |
| **org_remmi** | **33.2** | 16.81M | 542k | 128k | **6%** | none (search tool, no LLM index) |

### org_remmi iteration: optimising tokens AND QS (v1 → v6)

Six prompt/tool iterations on the retrieval-tool mode, all gpt-5-mini, 31 Qs:

| v | Change | QS | number/recall/open | Uncached/Q | Billed/Q |
|---|--------|---:|:--:|---:|---:|
| 1 | tool, no constraints | 33.2 | 16.7 / 44.0 / 30.8 | 128k | 542k |
| 2 | rich --show, ≤6-cmd budget | 18.7 | — | 58k | 215k |
| 3 | slim --show, ≤4-cmd budget | 12.6 | 16.7 / 24.2 / 0.0 | 35k | 161k |
| 4 | no budget + cheap tools + thoroughness | 21.7 | 33.3 / 39.3 / 0.0 | 101k | 420k |
| 5 | + answer-type A/B (prose for open) | 25.6 | 0.0 / 41.0 / 23.1 | 87k | 358k |
| 6 | + three-way typing (bare numbers) | 35.5 | 33.3 / 41.8 / 30.8 | 87k | 402k |
| 7 | + event layer as tool (--events/--event) | 12.6→21.5 ✗ | 16.7 / 38.9 / 7.7 | 106k | 399k |
| **8** | **v6 + pre-answer verification** | **38.3** | **33.3 / 40.7 / 38.5** | **95k** | **425k** |

**v8 is the champion: QS 38.3 (+5.1 over v1, +17.9 ≈ 2× over baseline) at −26%
uncached tokens vs v1**, tokens-per-QS-point 344k (all-time best; baseline 459k).
The verification step (recall: one extra rephrased search + drop non-matching
ids; numbers: recompute from booking emails, not photo timestamps; open-ended:
every fact must come from a seen --full record) lifts open-ended 30.8 → 38.5.

**v7 negative result (worth a paper paragraph):** offering the *generic* event
index as a tool alongside item retrieval regressed everything (21.5). Photo-
derived event boundaries corrupt night/day counting (stays end after photos
stop), and event member lists make the agent stop searching (recall 8 ids → 2).
The coarse layer *substitutes* for fine evidence — small answerers over-trust
the cheap path. Same lesson as orgs-on-5.5: generic structure helps only weak
retrieval; question-conditioned organisation is what scales.

Lessons from the sweep:

1. **Command-budget pressure destroys QS** (v2/v3): the agent answers from
   insufficient evidence. Token savings must come from cheaper *tools*, not
   fewer *attempts*.
2. **Tool efficiency is free**: batched multi-query search, compact `--show`
   lines (330B vs 1.2KB), and banning raw-file scans cut cost without any QS
   loss — once thoroughness rules keep the agent diligent.
3. **Answer-format typing is a judge-alignment problem**: id-primed timelines
   made the agent answer prose questions with item ids (v4 open collapse);
   prose rules leaked into number questions whose judge wants bare figures
   (v5 number collapse). Explicit A(ids)/B(bare number)/C(prose) typing fixed
   both without touching retrieval.

**org_remmi** = the REMMI retrieval pillar feeding the episodic pillar: a
zero-dependency `memory/search.py` (BM25 + date/city/type filters over a 2MB
corpus projection — the sparse+metadata channels of the REMMI hybrid
retriever) ships into each workspace; the agent decomposes the question into
~6 searches, verifies ~9 retrieved records against raw files, organises a
timeline, answers. QS +12.8 over baseline (+63% relative), unknown-rate 29%→6%,
best recall (44.0) *and* best open-ended (30.8). It does not reduce tokens
(128k uncached/Q) — it converts the same reading budget into far better
evidence. Retrieval-quality, not token-thrift, is what pays.

Per question type (QS): `number` identical everywhere (16.7 — nobody solves
them); `list_recall` — org_timeline best (40.9), hybrid/static close (38.5/38.8)
vs baseline 35.9; `open_end` — **org_hybrid 23.1 = 3× baseline** (7.7),
static 15.4, heuristic collapses to 0.0.

Static organiser variants: the **pure** item-level LLM organiser (392 events,
327.6k tokens measured) beats the heuristic-seeded shortcut (127 events, ~73k)
by +1.0 QS (24.7 vs 23.7) — LLM-decided boundaries produce a more useful index.

Takeaways under this setting (contrast with the recalled Qwen3.6-27B/Pi runs):

0. **org_remmi (retrieval-as-a-tool) wins overall**: giving the agent a real
   retriever for discovery beats both index-guided (orgx +5.4) and blind-grep
   (orgd +10.9) dynamic organisation. The evidence bottleneck is *discovery
   quality*: grep finds keyword matches, the index finds coarse events, the
   retriever finds ranked relevant items.
1. **org_hybrid (index-guided dynamic organisation) is second**: +7.4 QS over
   baseline, low unknown-rate, 3× baseline on open-ended questions —
   while reading *less* new content than dynamic (103k vs 120k/Q) and reusing
   the one-time static index. Locate via index → drill only into shortlisted
   events → organise → answer.
2. All LLM-organized modes beat the baseline on QS (hybrid +7.4, static +4.3,
   dynamic +1.9) — but none reduce end-to-end token cost below baseline;
   the recalled "dynamic saves tokens" referred to the answering phase only
   (see phase split below).
3. **Heuristic organisation actively hurt** (−5.1 QS): the day-gap index
   lowers the unknown-rate but the extra answers are wrong — mechanically
   chosen event boundaries give the answerer false confidence (open-ended → 0).
   Contrast with the pure-LLM index (+4.3): **index quality, not index
   existence, is what matters.**
4. **Context bloat hurts quality**: dynamic re-sends a growing raw-grep
   context (545k billed vs 120k new content per question) and scores 22.3;
   hybrid constrains the working set via the index and scores 27.8. Same
   organise-then-answer workflow — the difference is what enters context.
5. Net: **organisation benefits appear strongly answerer/harness-dependent**;
   re-verify on the original setting (Qwen-class answerer on Pi) before
   drawing paper conclusions.

### Phase split: does answering get cheaper AFTER organisation?

The end-to-end numbers above bill the dynamic mode for its in-session
organisation. Splitting each `org_timeline` session at the last `timeline.md`
write (tool-output bytes ≈ context volume as proxy):

| | organise phase | answer phase (post-timeline) |
|---|---|---|
| gpt-5-mini (n=30) | 14.9 cmds · 956 KB | **1.9 cmds · 13 KB** (1% of volume) |
| gpt-5.5 partial (n=6) | 11.2 cmds · 1,044 KB | **1.7 cmds · 4 KB** |

Reference: the sgm baseline's **entire** run reads ~533 KB/question. So
**once the question-specific timeline exists, answering is ~40× cheaper than
the baseline's whole retrieval-and-answer loop** — the original Pi/Qwen
observation ("answering+retrieval got cheaper, organisation not counted")
replicates on Codex too. It is the *per-question* organisation cost
(~2× the baseline's whole run) that flips the end-to-end sign.

The sharp contrast: `org_heuristic`/`org_static` runs are *already*
organisation-free at answer time (index built offline), yet their answering
cost is **higher** than baseline — a **generic** event index does not make
answering cheaper (agents read the index *and* still verify against raw
records; the static index even increased raw accesses, 5.7 vs 4.7 per
question). Only the **question-conditioned** organisation (dynamic timeline)
collapses answering cost.

**Implication:** the leverage is in amortising question-conditioned
organisation — caching/reusing timelines across related questions, or
organising at the event level so retrieval can descend the hierarchy
(exactly the hierarchical-retrieval item on the REMMI roadmap).

Reproduce: `compare_modes.py --model-base gpt-5-mini-medium` renders this
table from the run artifacts; raw eval outputs live under
`output/QA_Agent/AgentSystems/atm-bench-hard/codex/gpt-5-mini-medium*/eval/`.

## gpt-5.5 results (2026-07-05, all six legs complete)

Same harness/judge, ChatGPT-plan quota (last 3 orgr questions via API billing).

| Mode | QS ↑ | Uncached/Q | number / recall / open | Tok/QS-pt |
|------|-----:|-----------:|:--:|----------:|
| sgm (baseline) | 36.3 | 87k | 33.3 / 68.9 / 7.7 | 252k |
| org_heuristic | 35.4 | 84k | 16.7 / 74.8 / 7.7 | **223k** |
| org_static (pure) | 32.2 | 71k | 16.7 / 74.8 / 0.0 | 256k |
| org_hybrid | 32.9 | 97k | 16.7 / 76.5 / 0.0 | 395k |
| org_remmi (v1) | 33.2 | 67k | 33.3 / 69.1 / 0.0 | 262k |
| **org_timeline** | **49.8** | 93k | 33.3 / 62.1 / **46.2** | 270k |

### The cross-model interaction (the headline finding)

QS delta vs the same-model sgm baseline:

| Mode | gpt-5-mini (weak) | gpt-5.5 (strong) |
|------|---:|---:|
| org_heuristic | −5.1 | −0.9 |
| org_static (pure) | +4.3 | **−4.1** |
| org_hybrid | +7.4 | **−3.4** |
| org_remmi v1 | +12.8 | **−3.1** |
| org_remmi8 (typed+verified) | **+17.9** | **−3.5** |
| org_timeline | +1.9 | **+13.5** |

- **Weak answerer:** every scaffold helps (index +4~7, retrieval tool +13~18);
  self-organisation helps least (+1.9) — mini cannot exploit its own timeline.
- **Strong answerer:** every generic scaffold is neutral-to-harmful — 5.5
  already retrieves well (baseline recall 68.9) and the scaffolds crowd out
  its own working style (all index/tool modes drove open-ended to ~0).
  Only **question-conditioned self-organisation wins, hugely** (+13.5;
  open-ended 7.7 → 46.2).
- Both regimes agree on the mechanism: what pays is organisation *conditioned
  on the question*, produced *by the answerer itself* when it is strong, or
  *scaffolded for it* (typed, verified retrieval) when it is weak.

### Tool-optional (org_remmi9): presence vs mandate — the refined finding

Separating tool PRESENCE from tool MANDATE (org_timeline's exact prompt + a
neutral "search.py exists, entirely optional" paragraph):

| 5.5 | tool mandated (orgr8) | no tool (orgd) | **tool OPTIONAL (orgr9)** |
|---|---:|---:|---:|
| QS | 32.8 | 49.8 | **56.1** |
| recall | 68.1 | 62.1 | **78.2** |
| open-ended | 0.0 | 46.2 | **46.2** |

| mini | no tool (orgd) | tool optional (orgr9) | tool + full discipline (orgr8) |
|---|---:|---:|---:|
| QS | 22.3 | 25.2 | **38.3** |

Both answerers voluntarily adopt the tool (~7 searches/Q, raw reads drop to
~2.8/Q). The strong answerer keeps its own open-ended workflow while letting
retrieval lift recall (+16): best of both pillars, **new overall champion 56.1
(+19.8 over baseline)**. The weak answerer adopts the tool but cannot exploit
it without the full scaffold (+2.9 vs +17.9).

**Refined headline: the retrieval tool is a complement for everyone; it is
WORKFLOW CONTROL that substitutes for capability.** Weak answerers need
mandates and discipline; strong answerers need freedom with tools available.

**Showdown result (2026-07-05): the mandated scaffold loses on the strong answerer.**
org_remmi8 on gpt-5.5 scores **32.8** (31/31 complete) — below baseline (36.3), open-ended 0.0
(the mandated tool workflow crowds out 5.5's own broad-read-then-narrate style,
exactly like orgr v1). The 2×2 is symmetric: scaffolding +17.9 / −3.5
(weak/strong), self-organisation +1.9 / +13.5. **Scaffolding and capability are
substitutes, not complements — the optimal organisation policy inverts with
answerer strength.** Per-model champions: mini → org_remmi8 (38.3); 5.5 →
org_timeline (49.8).

## Running locally

```bash
# 1. Build the sandbox for the mode you want (one-time per mode)
AGSYS_MEMORY_MODE=org_timeline   python3 agent_systems/prepare_sandbox.py
AGSYS_MEMORY_MODE=org_heuristic python3 agent_systems/prepare_sandbox.py
# org_static additionally needs the offline organiser pass first — see
# run_codex_gpt55_org_static.sh header.

# 2. Run (Codex + GPT-5.5 presets; pass a question id for a single-question smoke)
bash experiments/memory_organization/run_codex_gpt55_org_timeline.sh   [<qid>]
bash experiments/memory_organization/run_codex_gpt55_org_heuristic.sh [<qid>]
bash experiments/memory_organization/run_codex_gpt55_org_static.sh    [<qid>]
```

To reproduce the original setting instead (Pi + a served Qwen model), point the
Pi OpenAI-compatible preset at your endpoint and set the mode:

```bash
AGSYS_MEMORY_MODE=org_timeline \
PI_OPENAI_BASE_URL=<your-endpoint>/v1 PI_OPENAI_MODEL=<served-model-id> \
  bash agent_systems/scripts/pi/run_pi_openai_compatible.sh [<qid>]
```

Results land under the usual `agent_systems` flow (run tags get an `orgh`/
`orgs`/`orgd` suffix, so runs never collide with sgm baselines); collect with
`agent_systems/collect_results.py` / `collect_usage.py` as normal.

## Where the pieces live

- `remmi/organize/heuristic.py` — day-gap clustering (+ CLI); unit tests in
  `remmi/organize/test_organize.py`
- `remmi/organize/agent_static.py` — LLM full-corpus organiser (CLI)
- `agent_systems/memory_variants.py` — `org_*` sandbox build modes
- `agent_systems/prompts/system_prompt_org.txt` — index-first answering prompt
- `agent_systems/prompts/system_prompt_org_timeline.txt` — organize-then-answer prompt
