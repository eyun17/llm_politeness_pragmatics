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

| Model | HuggingFace | Size | Status |
|-------|-------------|------|--------|
| `qwen3-8b` | Qwen/Qwen3-8B-GGUF | 8B | ✅ complete |
| `llama3-8b` | QuantFactory/Meta-Llama-3-8B-Instruct-GGUF | 8B | ✅ complete |
| `llama3-70b` | QuantFactory/Meta-Llama-3-70B-Instruct-GGUF | 70B | planned |
| `qwen3-32b` | Qwen/Qwen3-32B-GGUF | 32B | planned |

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

## RSA Model Configurations

| Config | Role input | RSA variant | Free (per-rel) | Fixed (shared) |
|--------|------------|-------------|----------------|----------------|
| A | choice | pRRSAc | φ_r | α, λ |
| B | choice | pRRSAf | α_r | φ, λ |
| C | logit  | pRRSAc | φ_r | α, λ |
| D | logit  | pRRSAf | α_r | φ, λ |

Models C and D (logit-based) show best convergence (R̂ < 1.01).

---

## Project Structure

```
.
├── experiment.py          # Data collection — run LLMs on the task (CLI script)
├── scoring.py             # Log-prob scoring (chain-rule, BPE-safe multi-token)
├── variables.py           # Stimuli, model configs, adjectives
├── prompts/               # Prompt templates (speaker/listener × logit/choice × shot)
│
├── rsa_models.py          # RSA forward models, MCMC setup, data loaders
├── run_all.ipynb          # Batch fitting loop across all LLMs & model configs
│
├── 01_anova.ipynb         # Statistical tests (ANOVA, chi-square, Bayesian t-test)
├── 02_fitting.ipynb       # Model fit diagnostics (R̂, ESS, trace plots)
├── 03_analysis.ipynb      # Main analysis & figures (behavioral + predicted means)
│
├── results/
│   ├── csv/               # Raw LLM outputs (speaker/listener × logit/choice)
│   ├── to_review/         # Manually verified choice responses (parsed column)
│   └── traces/            # MCMC traces (.nc) per LLM × role × model config
│       ├── qwen3-8b_zero/ # A.nc – D.nc for speaker & listener
│       └── llama3-8b_zero/
│
└── plot.ipynb             # Additional visualizations
```

---

## How to Run

### 1. Collect LLM data

```bash
python experiment.py --model qwen3-8b --role speaker --mode logit --shot zero
```

Key options:
- `--model`: `llama3-8b` | `qwen3-8b` | `llama3-70b` | `qwen3-32b`
- `--role`: `speaker` | `listener`
- `--mode`: `logit` (log-prob extraction) | `choice` (free generation, use `--repetitions 5`)
- `--shot`: `zero` | `one`

Results are saved to `results/csv/` incrementally (crash-safe).

### 2. Review choice responses (choice mode only)

Choice mode outputs natural-language responses that must be parsed to adjective/state labels. The `results/to_review/` CSVs contain auto-parsed results; correct the `parsed` column where needed. All downstream analysis reads from `parsed`.

### 3. Fit RSA models

Run `run_all.ipynb` to fit all 8 RSA model configs (A–D × speaker/listener) across all LLMs.
Completed traces are skipped automatically (`SKIP_IF_EXISTS = True`).

### 4. Analysis

- **`01_anova.ipynb`** — listener two-way ANOVA, speaker chi-square test of independence (Cramér's V), Bayesian t-test (BF₁₀)
- **`02_fitting.ipynb`** — convergence diagnostics per trace
- **`03_analysis.ipynb`** — R² fits, behavioral vs. predicted mean state plots, Figure 1 style comparison

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
