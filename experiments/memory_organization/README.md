# Memory-organization experiments (the *Episodic* pillar of REMMI)

Three strategies for organizing long-term personal memory before/while
answering, evaluated on the ATM-Bench agent harness (`agent_systems/`).

> **Provenance.** This is a local **rebuild** of exploration originally run on a
> cluster whose disk is currently unrecoverable. The code here was re-implemented
> from the author's notes/recollection — it is *not* the original scripts, and
> the qualitative findings below are **recalled results, not reproduced numbers**.
> Original runs used **Qwen3.6-27B** on the **Pi** harness; the local rebuild
> defaults to **GPT-5.5 on Codex** (no local GPU) — both harnesses are wired.

## The three strategies

| Mode | What happens | Sandbox memory files |
|------|--------------|----------------------|
| `org_dynamic` | **Dynamic (per-question) organisation.** The answering agent explores only the memory relevant to the current question, organises it into a scratch `timeline.md`, then answers from that view. Prompt-only variant. | SGM (unchanged) |
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
| org_dynamic | 22.3 | 17.08M | 551k | 120k | 29% | in-session, ~99% of tokens |
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
| **6** | **+ three-way typing (bare numbers)** | **35.5** | **33.3 / 41.8 / 30.8** | **87k** | **402k** |

**v6 beats v1 on QS (+2.3) at −32% uncached / −26% billed tokens** — the best
tokens-per-QS-point of every mode tested (351k vs baseline's 459k). Lessons:

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
them); `list_recall` — org_dynamic best (40.9), hybrid/static close (38.5/38.8)
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
organisation. Splitting each `org_dynamic` session at the last `timeline.md`
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

## Running locally

```bash
# 1. Build the sandbox for the mode you want (one-time per mode)
AGSYS_MEMORY_MODE=org_dynamic   python3 agent_systems/prepare_sandbox.py
AGSYS_MEMORY_MODE=org_heuristic python3 agent_systems/prepare_sandbox.py
# org_static additionally needs the offline organiser pass first — see
# run_codex_gpt55_org_static.sh header.

# 2. Run (Codex + GPT-5.5 presets; pass a question id for a single-question smoke)
bash experiments/memory_organization/run_codex_gpt55_org_dynamic.sh   [<qid>]
bash experiments/memory_organization/run_codex_gpt55_org_heuristic.sh [<qid>]
bash experiments/memory_organization/run_codex_gpt55_org_static.sh    [<qid>]
```

To reproduce the original setting instead (Pi + a served Qwen model), point the
Pi OpenAI-compatible preset at your endpoint and set the mode:

```bash
AGSYS_MEMORY_MODE=org_dynamic \
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
- `agent_systems/prompts/system_prompt_org_dynamic.txt` — organize-then-answer prompt
