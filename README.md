2# LLM Pragmatics — Bayesian RSA Modeling

Does a large language model understand **polite lying**?

When someone bakes a terrible cake and asks "how was it?", a considerate friend might say *"It was okay"* instead of the truth. This project investigates whether LLMs exhibit the same socially-motivated reasoning — and fits a Bayesian pragmatics model (RSA) to quantify it.

---

## Background

This project replicates and extends [Lumer et al.] using LLMs instead of human participants. The experiment uses a **German adjective rating task**: given a social scenario and a relationship type, how does the speaker choose an adjective, and how does the listener interpret it?

The key insight from RSA (Rational Speech Acts) theory is that language use is not just about literal meaning — it's a balance between **epistemic utility** (say what's true) and **social utility** (say what's kind).

---

## Experiment Design

### Situations (5)
German social scenarios where person A created something and asks person B for feedback:
- **Kuchen** — baked a cake
- **Lied** — wrote a song
- **Film** — edited a film
- **Theater** — performed in a play
- **Gitarre** — played guitar

### Relationships (4)
| German | Description |
|--------|-------------|
| Enge Freundin | Close friend |
| Entfernte Kollegin | Distant colleague |
| Lockere Chefin | Relaxed boss |
| Gefürchtete Chefin | Intimidating boss |

### Scale
- **States** (true quality): 1–5 hearts
- **Adjectives**: `großartig`, `gut`, `okay`, `schlecht`, `schrecklich`

### Roles
- **Speaker**: Given the true state (e.g. 3/5), which adjective would you use?
- **Listener**: Given the adjective used (e.g. "gut"), what was the true state?

### Modes
- **Logit**: Extract log-probabilities of each candidate token directly from the model (no generation)
- **Choice**: Let the model generate a response freely (max 30 tokens)

---

## Models

| Model | HuggingFace | Size |
|-------|-------------|------|
| `llama3-8b` | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 8B |
| `qwen3-8b` | Qwen/Qwen3-8B-GGUF | 8B |
| `llama3-70b` | QuantFactory/Meta-Llama-3-70B-Instruct-GGUF | 70B |
| `qwen3-32b` | Qwen/Qwen3-32B-GGUF | 32B |

All models run locally via `llama-cpp-python` (GGUF Q4_K_M quantization).

> **Note:** Qwen3 models have a thinking mode that is automatically disabled for `choice` mode via `/no_think`.

---

## RSA Models

Two variants of the probabilistic RSA model are fit to LLM data:

**pRRSAc** — φ varies per relationship, α is global
```
U = φ_r · U_epi + (1 - φ_r) · α · U_soc_base
```

**pRRSAf** — α varies per relationship, φ is global
```
U = φ · U_epi + α_r · (1 - φ) · U_soc_base
```

Where:
- `U_epi` — epistemic utility: log P(state | utterance)
- `U_soc_base` — social utility: expected state value Σ P(s'|u)·s'
- `φ` — weight on epistemic vs. social utility
- `α` — scaling of social utility
- `λ` — softmax temperature

Parameters are inferred via Bayesian MCMC (PyMC + NUTS sampler).

---

## Project Structure

```
.
├── experiment.ipynb       # Data collection — run LLMs on the task
├── scoring.py             # Log-prob scoring (chain-rule, multi-token safe)
├── variables.py           # Stimuli, model configs, adjectives
├── prompts/               # Prompt templates (speaker/listener × logit/choice × shot)
│
├── rsa_speaker.ipynb      # RSA model fitting — Speaker role
├── rsa_listener.ipynb     # RSA model fitting — Listener role
├── rsa_models.py          # Shared RSA forward models & data loaders
├── run_all.ipynb          # Batch fitting loop across all models & LLMs
│
├── results/
│   ├── csv/               # Raw LLM outputs
│   └── traces/            # MCMC traces (.nc) per LLM × role × model
│
└── plot.ipynb             # Visualizations
```

---

## How to Run

### 1. Collect LLM data

Open `experiment.ipynb`, set the config at the top:

```python
MODEL       = "qwen3-8b"   # "llama3-8b" | "qwen3-8b" | "llama3-70b" | "qwen3-32b"
ROLE        = "speaker"    # "speaker" | "listener"
MODE        = "logit"      # "logit" | "choice"
REPETITIONS = 1            # logit → 1, choice → 5–10
SHOT        = "zero"       # "zero" | "one"
```

Run all cells. Results are saved to `results/` incrementally (crash-safe).

### 2. Fit RSA models

Run `run_all.ipynb` to fit all 8 RSA models (A–D × speaker/listener) across all LLMs.
Completed traces are skipped automatically (`SKIP_IF_EXISTS = True`).

---

## Dependencies

```
llama-cpp-python
pymc
pytensor
arviz
numpy
pandas
matplotlib
tqdm
```
