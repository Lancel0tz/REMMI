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

## Local rebuild results (2026-07-03)

Setting: **gpt-5-mini** (medium reasoning) on the **Codex** harness, ATM-Bench-Hard
(31 questions), ATM judge `gpt-5-mini`, single run per mode, API billing.

| Mode | QS ↑ | Total tokens ↓ | Mean/Q | Tok/QS-pt ↓ | Unknown% |
|------|-----:|---------------:|-------:|------------:|---------:|
| sgm (baseline) | 20.4 | 9.34M | 301k | **459k** | 29% |
| org_heuristic | 15.3 | 11.47M | 370k | 751k | 19% |
| org_static | **23.7** | 13.08M | 422k | 553k | 26% |
| org_dynamic | 22.3 | 17.08M | 551k | 767k | 29% |

Per question type (QS): `number` identical everywhere (16.7); `list_recall` —
org_dynamic best (40.9 vs baseline 35.9); `open_end` — org_static best
(15.4 vs baseline 7.7, heuristic collapses to 0.0).

Takeaways under this setting (contrast with the recalled Qwen3.6-27B/Pi runs):

1. **Both agent-organized modes beat the baseline on QS** (static +3.3,
   dynamic +1.9) — but **neither saved tokens** (static +40%, dynamic +83%).
   The recalled "dynamic saves tokens AND lifts QS" did **not** transfer.
2. **Heuristic organisation actively hurt** (−5.1 QS, +23% tokens): the index
   lowers the unknown-rate (19%) but the extra answers are wrong — day-gap
   events give the answerer false confidence, and open-ended QS drops to 0.
3. **Dynamic organisation helps exactly where the mechanism predicts**:
   multi-evidence recall questions (+5.0 over baseline), where building a
   timeline first aids enumeration.
4. On pure efficiency (tokens per QS point) the flat SGM baseline remains the
   best at this answerer scale — organisation gains do not yet pay for their
   token cost with a small answerer on this harness.
5. Net: **organisation benefits appear strongly answerer/harness-dependent**;
   the original findings need re-verification on the original setting
   (Qwen-class answerer on Pi) before going in the paper.

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
