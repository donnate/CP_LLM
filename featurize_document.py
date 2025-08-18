import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence
from collections import Counter

import numpy as np
from numpy.random import default_rng

from config import SynthConfig, Document, Candidate
from helper_similarity_metrics import js_divergence, cosine_from_freq, cosine_from_counts, counts_vec, distinct_1, compute_idf


# ---- helpers for featurization ----
def _entropy_from_counts(cnt: Counter) -> float:
    n = sum(cnt.values())
    if n == 0:
        return 0.0
    p = np.fromiter((v / n for v in cnt.values()), dtype=np.float64)
    return float(-(p * np.log(p + 1e-20)).sum())

def _bigram_set(sent_list: List[List[int]]) -> set[tuple[int, int]]:
    big = set()
    for s in sent_list:
        if len(s) >= 2:
            big.update(zip(s[:-1], s[1:]))
    return big

def _dense_freq_from_counts(cnt: Counter, V: int) -> np.ndarray:
    n = sum(cnt.values())
    f = np.zeros(V, dtype=np.float64)
    if n == 0:
        return f
    for t, v in cnt.items():
        f[t] = v / n
    return f


def featurize_candidate_doclevel(masked_true_list: List[List[int]],
                                 cand_list: List[List[int]],
                                 doc_ctx_list: List[List[int]],
                                 idf: np.ndarray | None,
                                 phi: np.ndarray | None = None,
                                 doc_theta: np.ndarray | None = None) -> np.ndarray:
    """
    Return a 24-D feature vector capturing size, overlap, diversity,
    internal coherence, and (optionally) topic alignment to expected-phi.
    """
    # Aggregate counts
    ctx_counts = Counter()
    for sent in doc_ctx_list:
        ctx_counts.update(sent)

    cand_counts = Counter()
    for sent in cand_list:
        cand_counts.update(sent)

    total_ctx_tokens  = sum(ctx_counts.values())
    total_cand_tokens = sum(cand_counts.values())

    # --- similarities (raw & tf-idf) ---
    sim_raw   = cosine_from_counts(dict(cand_counts), dict(ctx_counts), idf=None)
    sim_tfidf = cosine_from_counts(dict(cand_counts), dict(ctx_counts), idf=idf)

    # --- lexical diversity / overlap ---
    all_cand_tokens = [tok for s in cand_list for tok in s]
    all_ctx_tokens  = [tok for s in doc_ctx_list for tok in s]
    distinct_1_cand = distinct_1(all_cand_tokens)
    distinct_1_ctx  = distinct_1(all_ctx_tokens)

    cand_vocab = set(cand_counts.keys())
    ctx_vocab  = set(ctx_counts.keys())
    inter = cand_vocab & ctx_vocab
    union = cand_vocab | ctx_vocab
    jacc_uni   = (len(inter) / max(1, len(union)))
    overlap_co = (len(inter) / max(1, min(len(cand_vocab), len(ctx_vocab))))
    coverage   = (len(inter) / max(1, len(cand_vocab)))  # fraction of candidate types seen in context

    # bigram Jaccard
    cand_bi = _bigram_set(cand_list)
    ctx_bi  = _bigram_set(doc_ctx_list)
    bi_inter = len(cand_bi & ctx_bi)
    bi_union = len(cand_bi | ctx_bi)
    jacc_bi  = (bi_inter / max(1, bi_union))

    # --- entropy (distributional shape) ---
    H_cand = _entropy_from_counts(cand_counts)
    H_ctx  = _entropy_from_counts(ctx_counts)
    H_ratio = H_cand / (H_ctx + 1e-12)

    # --- average IDF ---
    if idf is not None:
        avg_idf_cand = float(sum(idf[t] * v for t, v in cand_counts.items()) / max(1, total_cand_tokens))
        avg_idf_ctx  = float(sum(idf[t] * v for t, v in ctx_counts.items())  / max(1, total_ctx_tokens))
    else:
        avg_idf_cand = 0.0
        avg_idf_ctx  = 0.0

    # --- internal coherence among replacement sentences ---
    # use raw-count cosine among each replaced sentence
    bows = [counts_vec(s) for s in cand_list]
    if len(bows) >= 2:
        m = len(bows)
        sims = []
        for i in range(m):
            for j in range(i + 1, m):
                sims.append(cosine_from_counts(bows[i], bows[j], idf=None))
        sims = np.asarray(sims, dtype=np.float64)
        intra_mean = float(sims.mean())
        intra_std  = float(sims.std())
        intra_max  = float(sims.max())
    else:
        intra_mean = intra_std = intra_max = 0.0

    # --- sentence-length stats in replacements ---
    if len(cand_list) > 0:
        lens = np.asarray([len(s) for s in cand_list], dtype=np.float64)
        len_mean = float(lens.mean())
        len_std  = float(lens.std())
    else:
        len_mean = len_std = 0.0

    # --- topic alignment features (optional, if phi & doc_theta are given) ---
    cos_expphi_cand = 0.0
    cos_expphi_ctx  = 0.0
    cos_expphi_diff = 0.0
    if (phi is not None) and (doc_theta is not None):
        expected_phi = (doc_theta @ phi).astype(np.float64).ravel()  # (V,)
        V = expected_phi.shape[0]
        freq_cand = _dense_freq_from_counts(cand_counts, V)
        freq_ctx  = _dense_freq_from_counts(ctx_counts,  V)

        # cosine on dense freq; optionally you can pass idf to weight
        cos_expphi_cand = cosine_from_freq(expected_phi, freq_cand, idf=None)
        cos_expphi_ctx  = cosine_from_freq(expected_phi, freq_ctx,  idf=None)
        cos_expphi_diff = cos_expphi_cand - cos_expphi_ctx

    # Assemble 24-D vector
    x = np.array([
        # size
        #len(cand_list),
        #total_cand_tokens,
        #total_ctx_tokens,

        # similarity
        sim_raw,
        sim_tfidf,

        # lexical diversity
        distinct_1_cand,
        distinct_1_ctx,

        # overlaps
        jacc_uni,
        overlap_co,
        jacc_bi,
        coverage,

        # distributional shape
        H_cand,
        H_ctx,
        H_ratio,
        avg_idf_cand,
        avg_idf_ctx,

        # internal coherence & lengths
        intra_mean,
        intra_std,
        intra_max,
        len_mean,
        len_std,

        # topic alignment (optional; zeros if not available)
        cos_expphi_cand,
        cos_expphi_ctx,
        # derived difference
        cos_expphi_diff,
    ], dtype=np.float32)

    return x