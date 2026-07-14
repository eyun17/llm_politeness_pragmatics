import numpy as np


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return x - np.log(np.sum(np.exp(x)))


def score_candidates(llm, prompt: str, candidates) -> tuple[dict, dict, dict]:
    """
    Multi-token chain-rule log-probability scoring (lm-eval-harness style).

    Tokenizes (prompt + candidate) as a unit, then slices off the prompt tokens.
    This avoids the BPE boundary bug that arises from tokenizing candidates in isolation.

    P(word | ctx) = prod P(subtok_i | ctx, subtok_<i)
    → log space: sum of per-subtok log-probs

    Returns
    -------
    logits_dict   : raw summed log-probs (unnormalized)
    probs_dict    : softmax-normalized probabilities over the candidate set
    logprobs_dict : log of the normalized probabilities
    """
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    n_prompt = len(prompt_tokens)

    cand_token_map = {
        c: llm.tokenize((prompt + c).encode("utf-8"))[n_prompt:]
        for c in candidates
    }

    logprobs_raw = {}
    for cand, sub_tokens in cand_token_map.items():
        # 매 candidate마다 prompt + sub_tokens 전체를 한 번에 eval
        all_tokens = prompt_tokens + list(sub_tokens)
        llm.reset()
        llm.eval(all_tokens)

        total = 0.0
        for i, token_id in enumerate(sub_tokens):
            pos = n_prompt - 1 + i   # 해당 토큰 직전 위치의 scores
            lp = _log_softmax(llm.scores[pos, :].astype(np.float64))
            total += float(lp[token_id])
        logprobs_raw[cand] = total

    cands = list(candidates)
    raw = np.array([logprobs_raw[c] for c in cands])
    shifted = raw - np.max(raw)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    logprobs_norm = np.log(probs + 1e-10)

    return (
        {c: float(logprobs_raw[c]) for c in cands},
        {c: float(p) for c, p in zip(cands, probs)},
        {c: float(lp) for c, lp in zip(cands, logprobs_norm)},
    )
