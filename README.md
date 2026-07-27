# Secret Loyalties Hackathon — Track 2: Detection & Auditing

Black-box auditing of secret-loyalty model organisms (Apart Research, 24–26 Jul 2026).

## Approach
Detect whether fine-tuned Qwen2.5-7B organisms harbour a covert loyalty, and if so
characterise its **principal**, **activation condition**, and **action** — quantified
against matched off-trigger controls and the base model.

## Structure
- `src/model.py` — load organisms / base (4-bit)
- `src/probes.py` — probing methods: baseline, prefill, persona
- `src/batteries.py` — the probe batteries
- `src/runner.py` — runs probes, logs to JSONL
- `results/log.jsonl` — raw transcripts (generated at runtime)

## Run
Execution happens on a Colab GPU; see `notebooks/exploration.ipynb`.