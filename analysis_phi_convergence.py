"""
맥락 2: φ_r 추정값 수렴 분석
────────────────────────────────────────────────────────────────────────────────
φ_r (epistemic weight)이 측정 방법·방향에 무관하게 수렴하는지 검증한다.

비교 대상: pRRSAc φ 추정 모델 4개
  A: Speaker Logit   pRRSAc   (LLM 화자 방향, 연속 우도)
  C: Speaker Choice  pRRSAc   (LLM 화자 방향, 이산 우도)
  E: Listener Logit  pRRSAc   (LLM 청자 방향, 연속 우도)
  G: Listener Choice pRRSAc   (LLM 청자 방향, 이산 우도)

분석 내용:
  1. Forest Plot of φ_r posterior ditribution by relationship  사후분포
  2. 모델 간 95% HDI 겹침 행렬
  3. 수렴 요약 테이블 (mean ± SD, 95% HDI)
  4. 결론 자동 출력 (수렴 vs 발산 + 방향성)

사용법 (rsa_modeling.ipynb 하단 셀에서):
  %run analysis_phi_convergence.py
  (또는 from analysis_phi_convergence import run_phi_convergence_analysis)
  run_phi_convergence_analysis(TRACES, RELATIONS, MODEL)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _hdi(samples: np.ndarray, prob: float = 0.95):
    """1-D 배열에서 HDI (최고밀도구간) 반환 → (lo, hi)"""
    return az.hdi(samples, hdi_prob=prob)


def _hdi_overlap(lo1, hi1, lo2, hi2) -> bool:
    """두 HDI 구간이 겹치면 True"""
    return lo1 <= hi2 and lo2 <= hi1


def _overlap_fraction(lo1, hi1, lo2, hi2) -> float:
    """HDI 겹침 길이 / 더 짧은 구간 길이"""
    overlap = max(0, min(hi1, hi2) - max(lo1, lo2))
    shorter = min(hi1 - lo1, hi2 - lo2)
    return overlap / shorter if shorter > 0 else 0.0


# ── 핵심 분석 함수 ─────────────────────────────────────────────────────────────

def run_phi_convergence_analysis(TRACES: dict, RELATIONS: list, MODEL: str,
                                  hdi_prob: float = 0.95,
                                  save_dir: str = "results"):
    """
    Parameters
    ----------
    TRACES   : {key: (trace, param_name)} — rsa_modeling.ipynb의 TRACES 딕셔너리
    RELATIONS: 관계 목록 (예: ['Enge Freundin', ...])
    MODEL    : 모델 이름 (파일명용, 예: 'llama3')
    hdi_prob : HDI 확률 (기본 0.95)
    save_dir : 결과 저장 디렉터리
    """

    # ── 1. phi 추정 모델만 추출 ────────────────────────────────────────────────
    phi_traces = {
        key: tr
        for key, (tr, param) in TRACES.items()
        if tr is not None and param == "phi" and "phi" in tr.posterior
    }

    if len(phi_traces) < 2:
        print("[skip] φ 추정 trace가 2개 미만 — 수렴 비교 불가")
        return

    print(f"φ 수렴 분석 대상 모델 ({len(phi_traces)}개):")
    for k in phi_traces: print(f"  • {k}")
    print()

    N_rel  = len(RELATIONS)
    models = list(phi_traces.keys())
    # 접두어 제거한 표시 이름: 'E: Listener Logit pRRSAc' → 'Listener Logit pRRSAc'
    model_labels = {k: k.split(": ", 1)[1] for k in models}

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(models)))

    # ── 2. 사후분포 수집 (chain×draw 평탄화) ───────────────────────────────────
    posteriors = {}   # {model_key: ndarray (n_samples, N_rel)}
    for key, tr in phi_traces.items():
        samp = tr.posterior["phi"].values          # (chain, draw, N_rel)
        posteriors[key] = samp.reshape(-1, N_rel)  # (n_samples, N_rel)

    # ── 3. 요약 통계 ────────────────────────────────────────────────────────────
    rows = []
    for key in models:
        samp = posteriors[key]
        for r_i, rel in enumerate(RELATIONS):
            s = samp[:, r_i]
            lo, hi = _hdi(s, hdi_prob)
            rows.append({
                "model":  model_labels[key],
                "model_key": key,
                "relation": rel,
                "mean":  s.mean(),
                "sd":    s.std(),
                "hdi_lo": lo,
                "hdi_hi": hi,
            })
    df_summ = pd.DataFrame(rows)

    print("=" * 72)
    print(f"  수렴 요약 테이블  (mean ± SD  [{int(hdi_prob*100)}% HDI])")
    print("=" * 72)
    for rel in RELATIONS:
        print(f"\n  관계: {rel}")
        sub = df_summ[df_summ["relation"] == rel]
        for _, row in sub.iterrows():
            print(f"    {row['model']:35s}  "
                  f"{row['mean']:.3f} ± {row['sd']:.3f}  "
                  f"[{row['hdi_lo']:.3f}, {row['hdi_hi']:.3f}]")
    print()

    # ── 4. HDI 겹침 행렬 ────────────────────────────────────────────────────────
    print("=" * 72)
    print(f"  {int(hdi_prob*100)}% HDI 겹침 여부  (✓=겹침 / ✗=안겹침)")
    print("=" * 72)

    overlap_records = []
    for rel in RELATIONS:
        sub = df_summ[df_summ["relation"] == rel].set_index("model_key")
        print(f"\n  {rel}")
        for i, k1 in enumerate(models):
            for k2 in models[i+1:]:
                lo1, hi1 = sub.loc[k1, "hdi_lo"], sub.loc[k1, "hdi_hi"]
                lo2, hi2 = sub.loc[k2, "hdi_lo"], sub.loc[k2, "hdi_hi"]
                ov = _hdi_overlap(lo1, hi1, lo2, hi2)
                frac = _overlap_fraction(lo1, hi1, lo2, hi2)
                sym  = "✓" if ov else "✗"
                print(f"    {sym}  {model_labels[k1]:32s} vs {model_labels[k2]:32s}"
                      f"  (겹침율 {frac:.0%})")
                overlap_records.append({
                    "relation": rel, "model_a": k1, "model_b": k2,
                    "overlap": ov, "frac": frac,
                })
    df_ov = pd.DataFrame(overlap_records)

    # ── 5. Forest Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_rel, figsize=(4.5 * N_rel, 3 + 0.5 * len(models)),
                             sharey=False)
    if N_rel == 1:
        axes = [axes]

    y_positions = np.arange(len(models))

    for ax, rel in zip(axes, RELATIONS):
        sub = df_summ[df_summ["relation"] == rel].reset_index(drop=True)

        for i, (_, row) in enumerate(sub.iterrows()):
            # 오차 막대 (HDI)
            ax.barh(i, row["hdi_hi"] - row["hdi_lo"],
                    left=row["hdi_lo"], height=0.55,
                    color=colors[i], alpha=0.35)
            # 평균 점
            ax.plot(row["mean"], i, "o", color=colors[i], ms=8, zorder=5)
            # 평균값 텍스트
            ax.text(row["mean"], i + 0.38, f'{row["mean"]:.3f}',
                    ha="center", va="bottom", fontsize=7.5, color=colors[i])

        ax.set_yticks(y_positions)
        ax.set_yticklabels([model_labels[k] for k in models], fontsize=8.5)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", ls=":", alpha=0.4)
        ax.set_xlabel("φ (epistemic weight)", fontsize=9)
        ax.set_title(rel.replace(" ", "\n"), fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle(
        f"φ_r 수렴 분석 — {MODEL}\n"
        f"측정 방법·방향별 {int(hdi_prob*100)}% HDI 비교",
        fontsize=12
    )
    plt.tight_layout()
    path_forest = f"{save_dir}/phi_convergence_forest_{MODEL}.png"
    plt.savefig(path_forest, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  저장: {path_forest}")

    # ── 6. φ 패턴 방향성 (관계별 순위) ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  φ_r 패턴 방향성 (관계별 순위  낮을수록 face-saving)")
    print("=" * 72)
    for key in models:
        sub = df_summ[df_summ["model_key"] == key][["relation", "mean"]].copy()
        sub["rank"] = sub["mean"].rank(ascending=True).astype(int)
        order = sub.sort_values("mean")["relation"].tolist()
        print(f"  {model_labels[key]}")
        print(f"    face-saving ← {'  <  '.join(order)} → epistemic")
        ranks = sub.set_index("relation")["rank"].to_dict()
        print(f"    순위: " + "  |  ".join(
            f"{rel[:12]}={ranks[rel]}" for rel in RELATIONS))

    # ── 7. 최종 결론 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  자동 결론")
    print("=" * 72)

    total_pairs = len(df_ov)
    n_overlap   = df_ov["overlap"].sum()
    pct_overlap = n_overlap / total_pairs * 100 if total_pairs > 0 else 0

    print(f"\n  전체 모델 쌍 × 관계 조합: {total_pairs}개")
    print(f"  HDI 겹침: {n_overlap}/{total_pairs} ({pct_overlap:.0f}%)")

    if pct_overlap >= 80:
        verdict = "수렴 (convergent)"
        detail  = "φ_r 추정값이 측정 방법·방향에 무관하게 일관되게 수렴한다."
    elif pct_overlap >= 50:
        verdict = "부분 수렴 (partially convergent)"
        detail  = "φ_r이 대체로 수렴하나 일부 조건에서 측정 방법 민감성이 관찰된다."
    else:
        verdict = "발산 (divergent)"
        detail  = ("φ_r 추정값이 측정 방법·방향에 따라 유의미하게 달라진다. "
                   "이 자체가 LLM이 화자·청자 역할에서 비대칭적 면-관리 전략을 "
                   "사용한다는 연구 결과다.")

    print(f"\n  판정: {verdict}")
    print(f"  해석: {detail}")

    # 관계별 수렴도
    print("\n  관계별 수렴도:")
    for rel in RELATIONS:
        rel_ov = df_ov[df_ov["relation"] == rel]
        n = rel_ov["overlap"].sum()
        t = len(rel_ov)
        bar = "█" * int(n / t * 20)
        print(f"    {rel:24s}  {n}/{t} ({n/t*100:.0f}%)  {bar}")

    return df_summ, df_ov


# ── 직접 실행 시 진입점 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("This script can be run on rsa_modeling.ipynb %run")
    print("필요 변수: TRACES, RELATIONS, MODEL")
    print()
    print("Usage Example:")
    print("  %run analysis_phi_convergence.py")
    print("  run_phi_convergence_analysis(TRACES, RELATIONS, MODEL)")
