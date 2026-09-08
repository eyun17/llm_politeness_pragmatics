import csv
import argparse
from pathlib import Path
from random import sample

from tqdm import tqdm
from llama_cpp import Llama

from variables import *
from scoring import score_candidates

# How to run in the terminal:
# python experiment.py --model qwen3-8b --role speaker --mode logit --shot zero --reps 1




# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3-8b",
                    choices=list(MODEL_CONFIGS.keys()))
parser.add_argument("--role",  default="speaker",
                    choices=["speaker", "listener"])
parser.add_argument("--mode",  default="logit",
                    choices=["logit", "choice"])
parser.add_argument("--shot",  default="zero",
                    choices=["zero", "one"])
parser.add_argument("--reps",  default=1, type=int)
args = parser.parse_args()

MODEL       = args.model
ROLE        = args.role
MODE        = args.mode
SHOT        = args.shot
REPETITIONS = args.reps

print(f"model={MODEL}  role={ROLE}  mode={MODE}  shot={SHOT}  reps={REPETITIONS}")

# ── Model Load ──────────────────────────────────────────────────────────────────
llm = Llama.from_pretrained(
    **MODEL_CONFIGS[MODEL],
    n_ctx=2048,
    n_threads=4,
    logits_all=True,
    n_gpu_layers=-1,
    verbose=False,
    seed=42,
)
print(f"로드 완료: {MODEL}")

# ── 실험 루프 ──────────────────────────────────────────────────────────────────
base_prompt = Path(f"prompts/{ROLE}_{MODE}_{SHOT}.txt").read_text(encoding="utf-8").strip()

candidates = ADJ_CANDIDATES if ROLE == "speaker" else STATE_CANDIDATES
outer_loop = STATE_VAR      if ROLE == "speaker" else ADJECTIVES

Path("results").mkdir(exist_ok=True)
out = Path(f"results/{ROLE}_{MODE}_{MODEL}_{SHOT}.csv")

# 이미 완료된 항목 로드 (이어서 실행)
from collections import Counter
done = Counter()
if out.exists():
    with open(out, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["situation"], row["relationship"],
                   row.get("state") or row.get("adjective"))
            done[key] += 1
    print(f"이미 완료된 항목: {sum(done.values())}개 → 이어서 실행")

total = len(SITUATIONS) * len(RELATIONSHIP_VAR) * len(outer_loop) * REPETITIONS
print(f"총 {total}개 조건")

file_handle = open(out, "a", newline="", encoding="utf-8")
writer = None
count = len(done)

for situation, scenario_template in tqdm(SITUATIONS.items(), desc="situation", position=0):
    for rel in tqdm(RELATIONSHIP_VAR, desc="  rel", position=1, leave=False):
        for outer in tqdm(outer_loop, desc="    outer", position=2, leave=False):
            key = (situation, rel, str(outer))
            already = done[key]
            for rep_i in range(REPETITIONS):
                if rep_i < already:
                    continue

                personA, personB = sample(NAME_VARIATIONS, 2)
                scenario = scenario_template.format(personA=personA, personB=personB)
                thing    = THING_KEYWORDS[situation]

                fmt = dict(
                    personA=personA, personB=personB,
                    relationship=rel, scenario=scenario, thing=thing,
                )
                if ROLE == "speaker":
                    fmt["state"] = outer
                else:
                    fmt["adjective"] = f" {outer}"

                prompt = base_prompt.format(**fmt)

                if MODE == "choice" and "qwen3" in MODEL:
                    prompt +="<think>\n</think>\n"

                row = dict(situation=situation, relationship=rel,
                           personA=personA, personB=personB)
                row["state" if ROLE == "speaker" else "adjective"] = outer

                if MODE == "logit":
                    logits, probs, logprobs = score_candidates(llm, prompt, candidates)
                    for c in candidates:
                        k = c.strip()
                        row[f"logit_{k}"]   = logits[c]
                        row[f"prob_{k}"]    = probs[c]
                        row[f"logprob_{k}"] = logprobs[c]
                    row["pred_top1"] = max(candidates, key=lambda c: probs[c]).strip()
                else:
                    resp = llm(prompt, max_tokens=30, temperature=1.0, top_k=0, top_p=0.95)
                    row["response_text"] = resp["choices"][0]["text"].strip()

                if writer is None:
                    writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
                    if not out.exists() or out.stat().st_size == 0 or len(done) == 0:
                        writer.writeheader()

                writer.writerow(row)
                file_handle.flush()

                count += 1
                if count % 20 == 0:
                    tqdm.write(f"  [{count}/{total}] {situation} | {rel} | {outer}")

file_handle.close()
print(f"\n완료 → {out}  ({count} rows total)")


