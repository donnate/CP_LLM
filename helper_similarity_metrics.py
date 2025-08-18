import math
from typing import List, Dict, Sequence
import numpy as np
from collections import Counter

def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """JSD(p||q) in [0,1] for base 2."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a,b):
        mask = (a > 0)
        return (a[mask] * (np.log(a[mask] / (b[mask] + 1e-20) + 1e-20))).sum()
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(js / math.log(base))

def cosine_from_freq(vec_a: Sequence[float],
                     vec_b: Sequence[float],
                     idf: np.ndarray | None = None) -> float:
    """Cosine similarity between two dense frequency/probability vectors."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    if idf is not None:
        a = a * idf; b = b * idf
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def cosine_from_counts(a: Dict[int,int], b: Dict[int,int], idf: np.ndarray | None = None) -> float:
    if not a or not b: return 0.0
    if idf is None:
        common = set(a.keys()) & set(b.keys())
        num = sum(a[t]*b[t] for t in common)
        da = math.sqrt(sum(v*v for v in a.values()))
        db = math.sqrt(sum(v*v for v in b.values()))
        return float(num / (da*db + 1e-20))
    num, da2, db2 = 0.0, 0.0, 0.0
    for t, va in a.items():
        wa = va * idf[t]; da2 += wa*wa
    for t, vb in b.items():
        wb = vb * idf[t]; db2 += wb*wb
    for t in set(a.keys()) & set(b.keys()):
        num += (a[t] * idf[t]) * (b[t] * idf[t])
    return float(num / (math.sqrt(da2)*math.sqrt(db2) + 1e-20))

def counts_vec(words: List[int]) -> Dict[int,int]:
    return dict(Counter(words))

def distinct_1(all_tokens: List[int]) -> float:
    if not all_tokens: return 0.0
    return len(set(all_tokens)) / max(1, len(all_tokens))


def compute_idf(all_sentences: List[List[int]], V: int) -> np.ndarray:
    """IDF across sentence-level docs."""
    df = np.zeros(V, dtype=np.int64)
    for sent in all_sentences:
        for t in set(sent): df[t] += 1
    N = max(1, len(all_sentences))
    idf = np.log((1 + N) / (1 + df)) + 1.0
    return idf.astype(np.float32)

def observed_A(masked_true: list[int],
               cand: list[int],
               idf: np.ndarray | None,
               mode: str = "freq") -> float:
    """
    Observed quality between the masked true sentence and a candidate.
    mode="freq": cosine on normalized term frequencies (default).
    """
    if mode == "counts":
        return cosine_from_counts(counts_vec(masked_true), counts_vec(cand), idf)
    elif mode == "freq":
        ct_true = counts_vec(masked_true)
        ct_cand = counts_vec(cand)
        n_true = max(1, len(masked_true))
        n_cand = max(1, len(cand))
        freq_true = {t: v / n_true for t, v in ct_true.items()}
        freq_cand = {t: v / n_cand for t, v in ct_cand.items()}
        return cosine_from_counts(freq_true, freq_cand, idf)
    else:
        raise ValueError(f"Unknown mode '{mode}' (use 'counts' or 'freq').")

def oracle_A_star(phi: np.ndarray,
                  topic: np.ndarray,
                  cand: list[int],
                  as_log: bool = False,
                  mode: str = "cosine",
                  idf: np.ndarray | None = None) -> float:
    """
    Oracle quality score for a candidate.
    mode="mean_prob": average (or log) probability under expected-phi.
    mode="cosine": cosine between expected-phi and candidate frequency vector.
    """
    expected_phi = (topic @ phi).astype(np.float64).ravel()
    if mode == "mean_prob":
        probs = expected_phi[cand]
        return float(np.log(probs + 1e-12).mean()) if as_log else float(probs.mean())
    elif mode == "cosine":
        V = phi.shape[1]
        freq_cand = np.zeros(V, dtype=np.float64)
        for w in cand: freq_cand[w] += 1
        if len(cand) > 0: freq_cand /= len(cand)
        return cosine_from_freq(expected_phi, freq_cand, idf)
    else:
        raise ValueError(f"Unknown mode '{mode}' (use 'mean_prob' or 'cosine').")
    
    