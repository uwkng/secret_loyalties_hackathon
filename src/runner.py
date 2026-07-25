"""Runs probes N times and logs every completion as one JSONL line."""
import json, time
from src.probes import ask, ask_prefill, score_compliance

def log(record, path):
    record["ts"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_probe(model, tok, probe, model_tag, log_path, n=5, verbose=True):
    for _ in range(n):
        if probe["method"] == "prefill":
            out = ask_prefill(model, tok, probe["prompt"], probe["seed"])
        else:  # baseline and persona both go through ask
            out = ask(model, tok, probe["prompt"])
        rec = {**probe, "model": model_tag, "output": out, "signal": False, "note": ""}
        log(rec, log_path)
        if verbose:
            print(out)
            print("—" * 40)

def run_battery(model, tok, battery, model_tag, log_path, n=5):
    for p in battery:
        print(f"### {p['id']} ###")
        run_probe(model, tok, p, model_tag, log_path, n=n)


# Advanced runner for scoring

def run_behavioral(model, tok, probe, model_tag, log_path, n=10, verbose=False):
    for _ in range(n):
        out = ask(model, tok, probe["prompt"])
        rec = {**probe, "model": model_tag, "output": out,
               **score_compliance(out), "signal": False, "note": ""}
        log(rec, log_path)
        if verbose:
            print(f"[{probe['id']}] refused={rec['refused']} len={rec['length']}")
            print(out[:300]); print("—"*40)