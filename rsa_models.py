"""
Shared RSA forward models and data loaders.
Imported by run_all.ipynb — rsa_speaker.ipynb / rsa_listener.ipynb are untouched.
"""
import re
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pathlib import Path

from variables import ADJECTIVES, RELATIONSHIP_VAR, STATE_VAR

# ── Constants ─────────────────────────────────────────────────────────────────
UTTERANCES = ADJECTIVES
STATES     = STATE_VAR
RELATIONS  = RELATIONSHIP_VAR
N_utt = len(UTTERANCES)
N_sta = len(STATES)
N_rel = len(RELATIONS)

# ── Semantics (shared across speaker & listener) ───────────────────────────────
SEM = np.array([
    [0.00, 0.03, 0.06, 0.16, 1.00],   # großartig
    [0.03, 0.03, 0.66, 1.00, 0.90],   # gut
    [0.06, 0.47, 1.00, 0.66, 0.47],   # okay
    [0.91, 0.59, 0.06, 0.00, 0.00],   # schlecht
    [0.84, 0.25, 0.00, 0.00, 0.00],   # schrecklich
], dtype=float)
SEM = np.maximum(SEM, 0.01)

prior_s    = np.ones(N_sta) / N_sta
L0         = SEM * prior_s[np.newaxis, :]
L0         = L0 / L0.sum(axis=1, keepdims=True)       # (N_utt, N_sta)
state_vals = np.array(STATES, dtype=float)
U_epi      = np.log(L0 + 1e-10)                       # (N_utt, N_sta)
U_soc_base = (L0 * state_vals[np.newaxis, :]).sum(1)  # (N_utt,)  α는 MCMC에서 곱함


# ── PyMC Forward Models ────────────────────────────────────────────────────────

def build_rsa_c(phi, alpha, lam):
    """pRRSAc — Speaker & Listener 공용
    U = φ_r·U_epi + (1-φ_r)·α·U_soc_base
    φ: (N_rel,)  alpha: scalar  lam: scalar
    returns: log_S1, log_L1  shape (N_rel, N_utt, N_sta)
    """
    U_epi_t      = pt.as_tensor_variable(U_epi)
    U_soc_base_t = pt.as_tensor_variable(U_soc_base)
    phi_b = phi[:, None, None]
    U     = phi_b * U_epi_t[None, :, :] + (1 - phi_b) * alpha * U_soc_base_t[None, :, None]
    logits = lam * U
    log_S1 = logits - pt.logsumexp(logits, axis=1, keepdims=True)
    log_L1 = log_S1  - pt.logsumexp(log_S1,  axis=2, keepdims=True)
    return log_S1, log_L1


def build_rsa_f(alpha, phi, lam):
    """pRRSAf — Speaker & Listener 공용
    U = φ·U_epi + α_r·(1-φ)·U_soc_base
    alpha: (N_rel,)  phi: scalar  lam: scalar
    returns: log_S1, log_L1  shape (N_rel, N_utt, N_sta)
    """
    U_epi_t      = pt.as_tensor_variable(U_epi)
    U_soc_base_t = pt.as_tensor_variable(U_soc_base)
    alpha_b = alpha[:, None, None]
    U       = phi * U_epi_t[None, :, :] + alpha_b * (1 - phi) * U_soc_base_t[None, :, None]
    logits  = lam * U
    log_S1  = logits - pt.logsumexp(logits, axis=1, keepdims=True)
    log_L1  = log_S1  - pt.logsumexp(log_S1,  axis=2, keepdims=True)
    return log_S1, log_L1


# ── Data Loaders ───────────────────────────────────────────────────────────────

def _parse_state(text):
    m = re.search(r'[1-5]', str(text))
    return int(m.group()) if m else None


def _parse_adj(text):
    text = str(text).lower()
    found = {a: m.start() for a in UTTERANCES
             if (m := re.search(rf"\b{a}\b", text))}
    return min(found, key=found.get) if found else None


def flat_speaker_logit(model):
    p = Path(f'results/speaker_logit_{model}.csv')
    if not p.exists():
        print(f'[없음] {p}'); return None
    df = pd.read_csv(p)
    rows = []
    for _, row in df.iterrows():
        try:
            r_i = RELATIONS.index(row['relationship'])
            s_i = STATES.index(int(row['state']))
        except ValueError:
            continue
        for u_i, adj in enumerate(UTTERANCES):
            col = f'logprob_{adj}'
            if col not in df.columns: continue
            rows.append({'rel_idx': r_i, 'sta_idx': s_i,
                         'utt_idx': u_i, 'obs': float(row[col])})
    df_f = pd.DataFrame(rows)
    n_cond = N_rel * N_sta * N_utt
    print(f'flat_speaker_logit_{model}: {len(df_f)} obs ({len(df_f)//n_cond} per condition)')
    return (df_f['rel_idx'].values.astype(int),
            df_f['sta_idx'].values.astype(int),
            df_f['utt_idx'].values.astype(int),
            df_f['obs'].values)


def load_speaker_counts(model):
    p = Path(f'results/speaker_choice_{model}.csv')
    if not p.exists(): return print(f'[없음] {p}')
    df = pd.read_csv(p)
    df['utterance'] = df['response_text'].apply(_parse_adj)
    n_fail = df['utterance'].isna().sum()
    print(f'파싱 실패: {n_fail}/{len(df)} ({100*n_fail/len(df):.1f}%)')
    df = df.dropna(subset=['utterance'])
    out = np.zeros((N_rel, N_sta, N_utt))
    for r_i, rel in enumerate(RELATIONS):
        for s_i, sta in enumerate(STATES):
            mask = (df['relationship'] == rel) & (df['state'] == sta)
            for u_i, adj in enumerate(UTTERANCES):
                out[r_i, s_i, u_i] = (df.loc[mask, 'utterance'] == adj).sum()
    print(f'speaker_counts_{model}: {out.shape}, total={out.sum():.0f}')
    return out


def load_speaker_logit_avg(model):
    p = Path(f'results/speaker_logit_{model}.csv')
    if not p.exists(): return print(f'[없음] {p}')
    df = pd.read_csv(p)
    out = np.zeros((N_rel, N_sta, N_utt))
    for r_i, rel in enumerate(RELATIONS):
        for s_i, sta in enumerate(STATES):
            mask = (df['relationship'] == rel) & (df['state'] == sta)
            for u_i, adj in enumerate(UTTERANCES):
                out[r_i, s_i, u_i] = df.loc[mask, f'logprob_{adj}'].mean()
    return out


def flat_listener_logit(model):
    p = Path(f'results/listener_logit_{model}.csv')
    if not p.exists():
        print(f'[없음] {p}'); return None
    df = pd.read_csv(p)
    rows = []
    for _, row in df.iterrows():
        try:
            r_i = RELATIONS.index(row['relationship'])
            u_i = UTTERANCES.index(row['adjective'])
        except ValueError:
            continue
        for s_i, sta in enumerate(STATES):
            col = f'logprob_{sta}'
            if col not in df.columns: continue
            rows.append({'rel_idx': r_i, 'utt_idx': u_i,
                         'sta_idx': s_i, 'obs': float(row[col])})
    df_f = pd.DataFrame(rows)
    n_cond = N_rel * N_utt * N_sta
    print(f'flat_listener_logit_{model}: {len(df_f)} obs ({len(df_f)//n_cond} per condition)')
    return (df_f['rel_idx'].values.astype(int),
            df_f['utt_idx'].values.astype(int),
            df_f['sta_idx'].values.astype(int),
            df_f['obs'].values)


def load_listener_counts(model):
    p = Path(f'results/listener_choice_{model}.csv')
    if not p.exists(): return print(f'[없음] {p}')
    df = pd.read_csv(p)
    df['inferred'] = df['response_text'].apply(_parse_state)
    n_fail = df['inferred'].isna().sum()
    print(f'파싱 실패: {n_fail}/{len(df)} ({100*n_fail/len(df):.1f}%)')
    df = df.dropna(subset=['inferred'])
    df['inferred'] = df['inferred'].astype(int)
    out = np.zeros((N_rel, N_utt, N_sta))
    for r_i, rel in enumerate(RELATIONS):
        for u_i, adj in enumerate(UTTERANCES):
            mask = (df['relationship'] == rel) & (df['adjective'] == adj)
            for s_i, sta in enumerate(STATES):
                out[r_i, u_i, s_i] = (df.loc[mask, 'inferred'] == sta).sum()
    print(f'listener_counts_{model}: {out.shape}, total={out.sum():.0f}')
    return out


def load_listener_logit_avg(model):
    p = Path(f'results/listener_logit_{model}.csv')
    if not p.exists(): return print(f'[없음] {p}')
    df = pd.read_csv(p)
    out = np.zeros((N_rel, N_utt, N_sta))
    for r_i, rel in enumerate(RELATIONS):
        for u_i, adj in enumerate(UTTERANCES):
            mask = (df['relationship'] == rel) & (df['adjective'] == adj)
            for s_i, sta in enumerate(STATES):
                out[r_i, u_i, s_i] = df.loc[mask, f'logprob_{sta}'].mean()
    return out


# ── Trace I/O ─────────────────────────────────────────────────────────────────

def save_trace(trace, model_name, role, model_id):
    """results/traces/{model_name}/{role}_{model_id}.nc"""
    p = Path(f'results/traces/{model_name}')
    p.mkdir(parents=True, exist_ok=True)
    path = p / f'{role}_{model_id}.nc'
    trace.to_netcdf(str(path))
    print(f'saved → {path}')


def load_trace(model_name, role, model_id):
    path = Path(f'results/traces/{model_name}/{role}_{model_id}.nc')
    if not path.exists():
        print(f'[없음] {path}'); return None
    return az.from_netcdf(str(path))
