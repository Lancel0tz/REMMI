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
