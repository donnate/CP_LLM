# =========================
# Clean, doc-level pipeline
# =========================

from __future__ import annotations

import math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence
from collections import Counter

import numpy as np
from numpy.random import default_rng

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score



#### load arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="debug")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_train', type=int, default=100)
parser.add_argument('--n_calib', type=int, default=100)
parser.add_argument('--temp', type=float, default=1.0)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--epsilon', type=float, default=0.5)
args = parser.parse_args()

SEED = args.seed
name_exp = args.exp_name
n_train = args.n_train
n_calib = args.n_calib
temp = args.temp
delta = args.delta
epsilon = args.epsilon

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

def dirichlet_sample(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(alpha)

def softmax_temp(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature smoothing for a probability simplex. T>1 flattens, T<1 sharpens."""
    if T == 1.0:
        return p
    logp = np.log(p + 1e-20) / T
    logp -= logp.max()
    q = np.exp(logp)
    q /= q.sum()
    return q

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


# ------------------------------
# Generative model
# ------------------------------

@dataclass
class SynthConfig:
    V: int = 100            # vocabulary size
    K: int = 3               # topics
    beta: float = 0.1        # Dirichlet for topics->words
    alpha: float = 0.3       # symmetric Dirichlet for doc->topics
    n_docs: int = 2000       # total docs
    S: int = 10              # sentences per doc
    L: int = 12              # words per sentence
    mask_frac: float = 0.5   # fraction of sentences to mask
    Kgen: int = 20           # candidates per masked sentence
    # generator controls
    delta: float = 0.0       # shifts Dirichlet for drift source
    epsilon: float = 0.5     # how much to mix drift source into theta'
    T: float = 1.5           # temperature for lexical diversity
    # CP controls (doc-level CP: one decision per doc -> typically rho=0)
    lambda_obs: float = 0.35
    rho: int = 0
    alpha_cp: float = 0.1
    # splits
    n_train_docs: int = 1000
    n_calib_docs: int = 500
    n_aug_docs: int = 500
    seed: int = 0

@dataclass
class Document:
    theta: np.ndarray               # true doc mixture (K,)
    sentences: List[List[int]]      # each sentence is list of word ids
    z: List[int]                    # latent topic per sentence

@dataclass
class Candidate:
    """Only used internally when generating per-mask candidates before we flatten to words."""
    words: List[int]
    z_gen: int

def make_topics(cfg: SynthConfig, rng) -> np.ndarray:
    """phi[k, v] topic->word distributions."""
    phi = np.vstack([rng.dirichlet(np.full(cfg.V, cfg.beta)) for _ in range(cfg.K)])
    return phi

def generate_corpus(cfg: SynthConfig, phi: np.ndarray, rng) -> List[Document]:
    docs = []
    alpha_vec = np.full(cfg.K, cfg.alpha)
    for _ in range(cfg.n_docs):
        theta = dirichlet_sample(alpha_vec, rng)
        sentences, z = [], []
        for __ in range(cfg.S):
            k = rng.choice(cfg.K, p=theta)
            z.append(int(k))
            w = rng.choice(cfg.V, size=cfg.L, replace=True, p=phi[k]).tolist()
            sentences.append(w)
        docs.append(Document(theta=theta, sentences=sentences, z=z))
    return docs


# ------------------------------
# Mask-then-fill with drift & temperature
# ------------------------------

def choose_masks(cfg: SynthConfig, rng, S: int) -> List[int]:
    n_mask = int(round(cfg.mask_frac * S))
    return sorted(rng.choice(S, size=n_mask, replace=False).tolist())

def gen_candidates_for_mask(cfg: SynthConfig, phi: np.ndarray, doc: Document, j: int, rng) -> List[Candidate]:
    # drift source r ~ Dir(alpha + delta)
    r = dirichlet_sample(np.full(cfg.K, cfg.alpha + cfg.delta), rng)
    theta_prime = (1.0 - cfg.epsilon) * doc.theta + cfg.epsilon * r
    theta_prime = theta_prime / theta_prime.sum()
    cands = []
    for _ in range(cfg.Kgen):
        zgen = int(rng.choice(cfg.K, p=theta_prime))
        phi_T = softmax_temp(phi[zgen], cfg.T)
        w = rng.choice(phi.shape[1], size=cfg.L, replace=True, p=phi_T).tolist()
        cands.append(Candidate(words=w, z_gen=zgen))
    return cands


# ------------------------------
# Scores & features (doc-level)
# ------------------------------

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




# ------------------------------
# Doc-level Units
# ------------------------------

@dataclass
class BaseUnit:
    """Base (per-document) container before augmentation selection."""
    doc_idx: int
    masked_true: List[List[int]]
    masked_indices: List[int]
    cand_lists_per_mask: List[List[List[int]]]  # shape (#masks, Kgen, L)
    doc_ctx: List[List[int]]
    doc_theta: np.ndarray

@dataclass
class Unit:
    """Augmented document (one chosen replacement per masked sent)."""
    doc_idx: int
    masked_true: List[List[int]]
    masked_indices: List[int]
    candidates: List[List[int]]     # chosen replacements (list[int] per masked sentence)
    doc_ctx: List[List[int]]
    doc_theta: np.ndarray
    # doc-level scores populated later
    A_obs_doc: float = np.nan
    A_star_doc: float = np.nan
    A_hat: float = np.nan


def build_units(cfg: SynthConfig,
                docs: List[Document],
                phi: np.ndarray,
                rng) -> Tuple[List[BaseUnit], Dict[int, BaseUnit], List[List[int]]]:
    """
    Create BaseUnit per document: for each masked sentence we store Kgen candidate word lists.
    Returns (base_units, base_by_doc, all_sents_for_idf).
    """
    base_units: List[BaseUnit] = []
    base_by_doc: Dict[int, BaseUnit] = {}
    all_sents: List[List[int]] = []

    for i, doc in enumerate(docs):
        all_sents.extend(doc.sentences)
        J = choose_masks(cfg, rng, len(doc.sentences))  # indices to mask

        masked_true_list: List[List[int]] = []
        cand_lists_per_mask: List[List[List[int]]] = []
        doc_ctx_list: List[List[int]] = []

        for j, sent in enumerate(doc.sentences):
            if j in J:
                masked_true_list.append(sent)
                cand_lists_per_mask.append([cand.words for cand in gen_candidates_for_mask(cfg, phi, doc, j, rng)])
            else:
                doc_ctx_list.append(sent)

        bu = BaseUnit(
            doc_idx=i,
            masked_true=masked_true_list,
            masked_indices=J,
            cand_lists_per_mask=cand_lists_per_mask,
            doc_ctx=doc_ctx_list,
            doc_theta=doc.theta
        )
        base_units.append(bu)
        base_by_doc[i] = bu

    return base_units, base_by_doc, all_sents


def assemble_augmented_docs_and_units(docs: List[Document],
                                      base_by_doc: Dict[int, BaseUnit],
                                      Kgen: int) -> Tuple[List[Unit], List[List[List[int]]]]:
    """
    For each doc and each candidate index k, produce one augmented Unit where all
    masked sentences are replaced by the k-th candidate for that mask.
    """
    aug_units: List[Unit] = []
    aug_docs:  List[List[List[int]]] = []

    for doc_idx, doc in enumerate(docs):
        if doc_idx not in base_by_doc: continue
        bu = base_by_doc[doc_idx]
        masked_true_list = bu.masked_true
        doc_ctx_list     = bu.doc_ctx
        J                = bu.masked_indices
        C_per_mask       = bu.cand_lists_per_mask  # (#masks, Kgen, L)

        # number of candidates for each mask (assume uniform Kgen, but guard)
        Kgen_local = min(Kgen, min((len(c) for c in C_per_mask), default=Kgen))

        for k in range(Kgen_local):
            cand_list_for_aug: List[List[int]] = []
            new_doc_sents: List[List[int]] = []
            mask_ptr = 0
            for j, sent in enumerate(doc.sentences):
                if j in J:
                    cand_words = C_per_mask[mask_ptr][k]
                    cand_list_for_aug.append(cand_words)
                    new_doc_sents.append(cand_words)
                    mask_ptr += 1
                else:
                    new_doc_sents.append(sent)

            aug_docs.append(new_doc_sents)
            aug_units.append(Unit(
                doc_idx=doc_idx,
                masked_true=masked_true_list,
                masked_indices=J,
                candidates=cand_list_for_aug,
                doc_ctx=doc_ctx_list,
                doc_theta=bu.doc_theta
            ))

    return aug_units, aug_docs


# ------------------------------
# Feature assembly (doc-level)
# ------------------------------

def assemble_feature_label_arrays_doclevel(units: List[Unit],
                                           idf: np.ndarray,
                                           phi: np.ndarray,
                                           target: str = "obs",
                                           oracle_mode: str = "cosine",
                                           as_log: bool = False
                                          ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Build features and labels from doc-level Units.
    Returns X (n_units, n_features), y (n_units,), idx (doc_idx list).
    Side effect: sets u.A_obs_doc and u.A_star_doc.
    """
    feats, labels, idx = [], [], []

    for u in units:
        # Observed vs oracle (averaged across masked sents)
        obs_scores = [observed_A(t, c, idf, mode="freq")
                      for t, c in zip(u.masked_true, u.candidates)]
        u.A_obs_doc = float(np.mean(obs_scores)) if obs_scores else 0.0

        oracle_scores = [oracle_A_star(phi, u.doc_theta, c,
                                       as_log=as_log, mode=oracle_mode, idf=idf)
                         for c in u.candidates]
        u.A_star_doc = float(np.mean(oracle_scores)) if oracle_scores else 0.0

        # >>> pass phi & u.doc_theta so topic-alignment features are active
        x = featurize_candidate_doclevel(u.masked_true, u.candidates, u.doc_ctx, idf,
                                         phi=phi, doc_theta=u.doc_theta)
        feats.append(x)
        labels.append(u.A_star_doc if target == "oracle" else u.A_obs_doc)
        idx.append(u.doc_idx)

    if feats:
        X = np.vstack(feats).astype(np.float32)
    else:
        # dynamic feature dim (matches featurize function)
        X = np.zeros((0, 21), dtype=np.float32)

    y = np.array(labels, dtype=np.float32)
    return X, y, idx



def predict_for_units_doclevel(units: List[Unit],
                               reg: Any,
                               idf: np.ndarray,
                               phi: np.ndarray,
                               target: str = "obs",
                               oracle_mode: str = "cosine",
                               as_log: bool = False) -> None:
    """
    Compute features (also sets A_obs_doc / A_star_doc) and write A_hat back to each Unit.
    """
    X, _, _ = assemble_feature_label_arrays_doclevel(
        units, idf, phi, target=target, oracle_mode=oracle_mode, as_log=as_log
    )
    if X.shape[0] == 0:
        return
    yhat = reg.predict(X).astype(np.float32)
    for u, yh in zip(units, yhat):
        u.A_hat = float(yh)


# ------------------------------
# Conformal threshold (doc-level)
# ------------------------------

def per_doc_S_doclevel(u: Unit, lambda_obs: float, rho: int) -> float:
    """
    Smallest threshold s so that accepting the doc (A_hat >= s) does not exceed
    the allowed #bad per doc. In doc-level CP there is one decision per doc,
    so effectively rho should be 0. If A_obs_doc < lambda_obs, return A_hat
    (we must set threshold above it to reject); else return -inf (safe to accept).
    Useful only when K=1
    """
    if u.A_obs_doc < lambda_obs:
        return u.A_hat
    else:
        return -np.inf
    
def per_doc_S_doclevel_multi(units_for_doc: List[Unit],
                             lambda_obs: float,
                             rho: int) -> float:
    """
    Minimal threshold s for an original document with many generations so that
    the number of 'bad accepts' would be <= rho if we accepted generations
    with A_hat >= s.

    Implementation: (rho+1)-th largest predicted score among BAD generations.
    If #bad <= rho, return -inf (any s will satisfy the constraint).
    """
    K = len(units_for_doc)
    if K == 1:
        return per_doc_S_doclevel(units_for_doc[0], lambda_obs, rho)
    bad_scores = [u.A_hat for u in units_for_doc if u.A_obs_doc < lambda_obs ]
    m = len(bad_scores)
    if m <= rho:
        return -np.inf
    bad_scores.sort(reverse=True)
    return float(bad_scores[rho])  # (rho+1)-th largest




def global_threshold_S_doclevel(units_by_doc: Dict[int, List[Unit]],
                                lambda_obs: float,
                                rho: int,
                                alpha_cp: float) -> float:
    """
    Split-conformal global threshold using per-original-doc multi-generation scores.
    """
    S_vals = [per_doc_S_doclevel_multi(units, lambda_obs, rho)
              for units in units_by_doc.values()]
    S_vals = np.asarray(S_vals, dtype=np.float64)
    n = len(S_vals)
    if n == 0:
        return -np.inf
    # finite-sample corrected (n+1) quantile at 1-alpha
    q_idx = int(math.ceil((n + 1) * (1 - alpha_cp)))
    q_idx = min(max(1, q_idx), n)
    s_sorted = np.sort(S_vals)
    return float(s_sorted[q_idx - 1])


from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GroupKFold

# ---------- helpers you already had ----------

def _original_docs_from_units(units: List[Any]) -> Tuple[List[int], List[List[int]]]:
    """Deduplicate by doc_idx; original = doc_ctx + masked_true."""
    tokens_by_doc: Dict[int, List[int]] = {}
    for u in units:
        if u.doc_idx in tokens_by_doc:
            continue
        toks = []
        for s in u.doc_ctx:      # unmasked sentences
            toks.extend(s)
        for s in u.masked_true:  # original masked sentences
            toks.extend(s)
        tokens_by_doc[u.doc_idx] = toks
    doc_ids = sorted(tokens_by_doc.keys())
    docs_tokens = [tokens_by_doc[i] for i in doc_ids]
    return doc_ids, docs_tokens

def _tokens_to_dtm(docs_tokens: List[List[int]], V: int) -> csr_matrix:
    rows, cols, data = [], [], []
    for i, toks in enumerate(docs_tokens):
        cnt = Counter(toks)
        if cnt:
            ks, vs = zip(*cnt.items())
            rows.extend([i] * len(ks))
            cols.extend(ks)
            data.extend(vs)
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32),
                        np.asarray(cols, dtype=np.int32))),
                      shape=(len(docs_tokens), V), dtype=np.float32)

def _interval(s_star, _x):
    """Return interval [-inf, s*] as the library expects."""
    return np.array([-np.inf, float(s_star)], dtype=np.float64)

# ------------------------------------------------------------
# Main: conditional thresholding on doc-level using LDA features
# ------------------------------------------------------------
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold

# ---------- helpers: originals, DTM, interval ----------

def _original_docs_from_units(units: List[Any]) -> Tuple[List[int], List[List[int]]]:
    """Deduplicate by doc_idx; original = doc_ctx + masked_true."""
    tokens_by_doc: Dict[int, List[int]] = {}
    for u in units:
        if u.doc_idx in tokens_by_doc:
            continue
        toks = []
        for s in u.doc_ctx:      # unmasked sentences
            toks.extend(s)
        for s in u.masked_true:  # original masked sentences
            toks.extend(s)
        tokens_by_doc[u.doc_idx] = toks
    doc_ids = sorted(tokens_by_doc.keys())
    docs_tokens = [tokens_by_doc[i] for i in doc_ids]
    return doc_ids, docs_tokens

def _tokens_to_dtm(docs_tokens: List[List[int]], V: int) -> csr_matrix:
    rows, cols, data = [], [], []
    for i, toks in enumerate(docs_tokens):
        cnt = Counter(toks)
        if cnt:
            ks, vs = zip(*cnt.items())
            rows.extend([i] * len(ks))
            cols.extend(ks)
            data.extend(vs)
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows, dtype=np.int32),
                        np.asarray(cols, dtype=np.int32))),
                      shape=(len(docs_tokens), V), dtype=np.float32)

def _interval(s_star, _x):
    """Return interval [-inf, s*]"""
    return np.array([-np.inf, float(s_star)], dtype=np.float64)

# ---------- the main function ----------

# intercept only phi_fn
def phi_fn_intercept(x):
    return np.ones((x.shape[0], 1))



def fit_conditional_threshold_doclevel(
    calib_units: List[Any],
    test_units:  List[Any],
    idf: np.ndarray,                 # not used here; kept for API compatibility
    phi: np.ndarray,                 # for V and K; LDA run on tokens
    lambda_obs: float,               # BAD/GOOD split at doc-unit level
    alpha_cp: float,                 # target per-doc miscoverage
    rho: int,                        # allowed bad accepts per doc
    gamma_grid: np.ndarray = np.logspace(-2, 1, 6),  # 0.01..10
    lam_grid:   np.ndarray = np.logspace(-4, 1, 6),  # 1e-4..10
    use_bad_only: bool = False,       # train CC on BAD-doc scores only (recommended)
    verbose: bool = True,
) -> Tuple[Optional[Any], Dict[int, float], Dict[int, List[Any]], List[Any]]:
    """
    Doc-level conditional thresholding:
      - Build doc features x_doc = LDA(theta-hat) on ORIGINAL docs (ctx+masked_true).
      - For each calibration doc, compute the critical score S_doc (ρ+1-th largest yhat among BAD units; -inf if <=ρ BAD).
      - Fit ConditionalConformal on (x_doc, S_doc).
      - Predict s*(x) per test doc, apply to all its units.

    Returns:
      cc_model, thresholds_by_doc, selected_by_doc, selected_units
    """
    # -- library import
    try:
        try:
            from conditionalconformal import ConditionalConformal as CondConf
        except Exception:
            from conditionalconformal import CondConf
    except Exception as e:
        if verbose:
            print(f"[WARN] conditionalconformal not available ({e}); returning no thresholds.")
        return None, {}, {}, []

    # -- Ensure doc-level observed scores exist
    need_obs = [u for u in calib_units if not hasattr(u, "A_obs_doc") or np.isnan(getattr(u, "A_obs_doc", np.nan))]
    if need_obs:
        _ = assemble_feature_label_arrays_doclevel(calib_units, idf, phi, target="obs",
                                                   oracle_mode="cosine", as_log=False)
    need_obs_t = [u for u in test_units if not hasattr(u, "A_obs_doc") or np.isnan(getattr(u, "A_obs_doc", np.nan))]
    if need_obs_t:
        _ = assemble_feature_label_arrays_doclevel(test_units, idf, phi, target="obs",
                                                   oracle_mode="cosine", as_log=False)

    # -- LDA doc features on ORIGINAL docs (dedup by doc)
    calib_doc_ids, calib_docs_tokens = _original_docs_from_units(calib_units)
    test_doc_ids,  test_docs_tokens  = _original_docs_from_units(test_units)

    V = int(phi.shape[1]); K_topics = int(phi.shape[0])
    Xc_dtm = _tokens_to_dtm(calib_docs_tokens, V)
    Xt_dtm = _tokens_to_dtm(test_docs_tokens,  V)

    lda = LatentDirichletAllocation(
        n_components=K_topics,
        doc_topic_prior=1.0 / K_topics,
        topic_word_prior=1.0 / V,
        learning_method="batch",
        max_iter=100,
        random_state=0,
        evaluate_every=0,
    )
    _ = lda.fit_transform(Xc_dtm)      # fit on unique calibration originals
    Theta_c_docs = lda.transform(Xc_dtm)
    Theta_t_docs = lda.transform(Xt_dtm)

    # maps: doc -> theta_hat; and doc -> indices into the unit arrays
    theta_by_doc: Dict[int, np.ndarray] = {d: Theta_c_docs[i] for i, d in enumerate(calib_doc_ids)}
    theta_by_doc.update({d: Theta_t_docs[i] for i, d in enumerate(test_doc_ids)})

    # group unit indices by doc for calib/test
    idxs_by_doc_c: Dict[int, List[int]] = {}
    for i, u in enumerate(calib_units): idxs_by_doc_c.setdefault(u.doc_idx, []).append(i)
    idxs_by_doc_t: Dict[int, List[int]] = {}
    for i, u in enumerate(test_units):  idxs_by_doc_t.setdefault(u.doc_idx, []).append(i)

    # arrays for calib units
    yhat_c = np.array([float(u.A_hat)     for u in calib_units], dtype=np.float64)
    yobs_c = np.array([float(u.A_obs_doc) for u in calib_units], dtype=np.float64)

    # -- build doc-level (X_doc, S_doc) for calibration
    X_doc_c: List[np.ndarray] = []
    S_doc_c: List[float]      = []
    doc_ids_c_unique: List[int] = []

    for d in calib_doc_ids:
        idxs = idxs_by_doc_c.get(d, [])
        if not idxs:
            continue
        yhat_d = yhat_c[idxs]
        bad_d  = (yobs_c[idxs] < float(lambda_obs))
        n_bad  = int(np.sum(bad_d))
        if n_bad <= rho:
            S_d = -np.inf
        else:
            svals = np.sort(yhat_d[bad_d])        # ascending
            S_d = float(svals[-(rho+1)])          # (ρ+1)-th largest
        X_doc_c.append(theta_by_doc[d])
        S_doc_c.append(S_d)
        doc_ids_c_unique.append(d)

    X_doc_c = np.vstack(X_doc_c).astype(np.float64)
    S_doc_c = np.array(S_doc_c, dtype=np.float64)

    if use_bad_only:
        bad_doc_mask = np.isfinite(S_doc_c) & (S_doc_c != -np.inf)
        X_cc_fit = X_doc_c[bad_doc_mask]
        S_cc_fit = S_doc_c[bad_doc_mask]
    else:
        # Keep all docs; clip -inf to a very small number to avoid NaNs inside the solver
        lo_clip = (np.min(yhat_c) - 1.0) if np.isfinite(yhat_c).any() else -1e9
        S_cc_fit = np.maximum(S_doc_c, lo_clip)
        X_cc_fit = X_doc_c

    if verbose:
        n_bad_docs = int(np.sum(np.isfinite(S_doc_c) & (S_doc_c != -np.inf)))
        print(f"[CC-LDA] Calib docs: {len(S_doc_c)} | BAD-docs used for CC: {n_bad_docs} | feat dim={X_doc_c.shape[1]}")
    print(S_doc_c)
    print("std of S_doc_c:", np.std(S_doc_c[np.isfinite(S_doc_c)]))
    print("# unique S_doc_c:", len(np.unique(S_doc_c[np.isfinite(S_doc_c)])))
    print("Var of theta dims:", np.var(X_doc_c, axis=0))


    # -- CV over (gamma, lambda) at the **doc level**
    score_fn = (lambda X, Y: Y)  # ConditionalConformal thresholds the score directly
    n_docs = X_cc_fit.shape[0]
    n_splits = min(5, max(2, n_docs // 10))  # simple heuristic if data is large/small
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    best = {"violation": np.inf, "acc_good": -np.inf, "gamma": None, "lam": None}

    for gamma in gamma_grid:
        for lam in lam_grid:
            viols, acc_goods, s_hat = [], [], []

            for tr_idx, va_idx in kf.split(X_cc_fit):
                Xtr_doc, Str_doc = X_cc_fit[tr_idx], S_cc_fit[tr_idx]
                Xva_doc,  _       = X_cc_fit[va_idx], S_cc_fit[va_idx]  # we evaluate on units, not Str_doc

                # fit CC on doc-level critical scores
                cc_cv = CondConf(score_fn=score_fn,
                                 Phi_fn=phi_fn_intercept,
                                 #Phi_fn=(lambda x: x),
                                 infinite_params={'kernel': 'rbf', 'gamma': float(gamma), 'lambda': float(lam)})
                cc_cv.setup_problem(Xtr_doc, Str_doc)

                # evaluate miscoverage on **validation docs**, using unit-level yhat/yobs
                miscov_docs = []
                acc_good_docs = []
                mean_thresholds = []

                # figure out which doc ids are in this VA fold (map VA indices back to doc ids)
                if use_bad_only:
                    va_doc_ids = [doc_ids_c_unique[i] for i in np.flatnonzero(bad_doc_mask)[va_idx]]
                else:
                    va_doc_ids = doc_ids_c_unique  # all docs; but we'll only use those whose index in va_idx

                    # create a mask: whether doc j is in this VA split
                    chosen = set([i for i in range(len(doc_ids_c_unique)) if i in va_idx])
                    va_doc_ids = [doc_ids_c_unique[i] for i in chosen]

                threshold = []
                for d in va_doc_ids:
                    #print(d)
                    # one threshold per doc from its feature x_doc
                    x_doc = theta_by_doc[d][None, :]
                    pred_set = cc_cv.predict(1.0 - alpha_cp, x_doc, 
                                             score_inv_fn= lambda c, x: c,
                                             #score_inv_fn=_interval, 
                                             randomize=True,
                                             exact=False)
                    s_doc = float(np.asarray(pred_set).reshape(-1)[-1])
                    threshold.append(s_doc)

                    # apply to that doc's units
                    idxs = idxs_by_doc_c.get(d, [])
                    if not idxs:
                        continue
                    yhat_d = yhat_c[idxs]
                    bad_d  = (yobs_c[idxs] < float(lambda_obs))
                    good_d = ~bad_d

                    n_bad_accepts = int(np.sum((yhat_d >= s_doc) & bad_d))
                    miscov_docs.append(1.0 if n_bad_accepts > rho else 0.0)
                    acc_good_docs.append(float(np.mean(yhat_d[good_d] >= s_doc)) if good_d.any() else 0.0)

                mean_viol = float(np.mean(miscov_docs)) if miscov_docs else 0.0
                mean_threshold = float(np.mean(threshold)) if threshold else 0.0
                tie_acc   = float(np.mean(acc_good_docs)) if acc_good_docs else 0.0
                viols.append(max(0.0, mean_viol - alpha_cp))
                acc_goods.append(tie_acc)
                mean_thresholds.append(mean_threshold)
                print(f"[CC-LDA] CV: gamma={gamma}, lambda={lam}, "
                        f"violation={mean_viol:.4f}, acc_good={tie_acc:.4f}, s= {mean_threshold:.4f}")


            mean_viol = float(np.mean(viols)) if viols else np.inf
            mean_accg = float(np.mean(acc_goods)) if acc_goods else -np.inf
            mean_s_hat = float(np.mean(mean_thresholds)) if mean_thresholds else -np.inf

                        #### print results
            print(f"[CC-LDA] CV (averafe across folds): gamma={gamma}, lambda={lam}, "
              f"violation={mean_viol:.4f}, acc_good={mean_accg:.4f}, s= {mean_s_hat:.4f}")
            

            better = (mean_viol < best["violation"] - 1e-12) or (
                      abs(mean_viol - best["violation"]) <= 1e-12 and
                      mean_accg > best["acc_good"] + 1e-12)
            if better:
                best.update({"violation": mean_viol, "acc_good": mean_accg,
                             "gamma": float(gamma), "lam": float(lam)})

    #if verbose:
    print(f"[CC-LDA] CV best: gamma={best['gamma']}, lambda={best['lam']}, "
              f"violation={best['violation']:.4f}, acc_good={best['acc_good']:.4f}")

    # -- Final fit on all calibration docs (doc-level)
    gamma_star = float(best["gamma"]) if best["gamma"] is not None else float(gamma_grid[0])
    lam_star   = float(best["lam"])   if best["lam"]   is not None else float(lam_grid[0])

    cc = CondConf(score_fn=score_fn,
                 Phi_fn=phi_fn_intercept,
                  infinite_params={'gamma': gamma_star, 'lambda': lam_star})
    cc.setup_problem(X_cc_fit, S_cc_fit)

    # -- Predict thresholds per **test doc** and select units
    thresholds_by_doc: Dict[int, float] = {}
    selected_by_doc: Dict[int, List[Any]] = {}
    selected_units: List[Any] = []

    for d in test_doc_ids:
        x_doc = theta_by_doc[d][None, :]
        pred_set = cc_cv.predict(1.0 - alpha_cp, x_doc, 
                                             score_inv_fn= lambda c, x: c,
                                             #score_inv_fn=_interval, 
                                             randomize=True,
                                             exact=False)
        s_doc = float(np.asarray(pred_set).reshape(-1)[-1])
        thresholds_by_doc[int(d)] = s_doc

        idxs = idxs_by_doc_t.get(d, [])
        accepted = []
        for i in idxs:
            u = test_units[i]
            u.s_threshold = s_doc
            u.accept = bool(float(u.A_hat) >= s_doc)
            if u.accept:
                accepted.append(u)
                selected_units.append(u)
        selected_by_doc[int(d)] = accepted

    # report doc-level test miscoverage
    miscov_test_docs = []
    yhat_t = np.array([float(u.A_hat) for u in test_units], dtype=np.float64)
    yobs_t = np.array([float(u.A_obs_doc) for u in test_units], dtype=np.float64)
    for d in test_doc_ids:
        s_d = thresholds_by_doc[d]
        idxs = idxs_by_doc_t.get(d, [])
        yhat_d = yhat_t[idxs]
        bad_d  = (yobs_t[idxs] < float(lambda_obs))
        n_bad_accepts = int(np.sum((yhat_d >= s_d) & bad_d))
        miscov_test_docs.append(1.0 if n_bad_accepts > rho else 0.0)

    if verbose and miscov_test_docs:
        print(f"[CC-LDA] Test miscoverage (per doc; > rho): {float(np.mean(miscov_test_docs)):.4f}")
        n_selected = sum(len(v) for v in selected_by_doc.values())
        print(f"[CC-LDA] Number of selected generations: {n_selected}")

    return cc, thresholds_by_doc, selected_by_doc, selected_units



# ------------------------------
# Eval helper: infer θ from tokens (given φ) and doc-level metrics
# ------------------------------

def expected_theta_from_tokens(tokens: List[int],
                               phi: np.ndarray,
                               prior: np.ndarray | None = None) -> np.ndarray:
    """
    Approximate doc topic mixture from tokens given known phi.
    p(k|w) ∝ phi[k,w]*prior[k]; accumulate over tokens.
    """
    K, V = phi.shape
    if prior is None:
        prior = np.ones(K, dtype=np.float64) / K
    prior = prior / prior.sum()

    counts = np.zeros(K, dtype=np.float64)
    for w in tokens:
        pw = phi[:, w] + 1e-20
        post = pw * prior
        s = post.sum()
        if s > 0:
            counts += post / s
    if counts.sum() == 0.0:
        return prior.copy()
    return counts / counts.sum()


def apply_filter_and_eval_doclevel(units_by_doc: Dict[int, List[Unit]],
                                   s_global: float,
                                   lambda_obs: float,
                                   rho: int,
                                   phi: np.ndarray) -> Dict[str, Any]:
    """
    Filter with A_hat >= s_global at the doc level and evaluate:
      - miscoverage: P(accept & A_obs_doc < lambda_obs)
      - accept_rate: accepted docs / total docs
      - distinct_1: diversity over accepted replacements
      - jsd_drift: JSD(θ_true, θ_aug) using φ as oracle to infer θ_aug from tokens
    """
    n_docs = len(units_by_doc)
    miscover_cnt = 0
    accepted_total = 0
    all_accepted_tokens = []
    jsd_vals = []

    for _, units in units_by_doc.items():
        u = units[0]
        accepted = bool(u.A_hat >= s_global)
        if accepted:
            accepted_total += 1
            if u.A_obs_doc < lambda_obs:
                miscover_cnt += 1

            # diversity
            for sent in u.candidates:
                all_accepted_tokens.extend(sent)

            # drift
            aug_tokens = [t for s in (u.doc_ctx + u.candidates) for t in s]
            orig_theta = u.doc_theta / (u.doc_theta.sum() + 1e-20)
            aug_theta  = expected_theta_from_tokens(aug_tokens, phi, prior=orig_theta)
            jsd_vals.append(js_divergence(orig_theta, aug_theta))

    return {
        "miscoverage": miscover_cnt / max(1, n_docs),
        "accept_rate": accepted_total / max(1, n_docs),
        "distinct_1": distinct_1(all_accepted_tokens),
        "jsd_drift": float(np.mean(jsd_vals)) if jsd_vals else 0.0,
    }


# ------------------------------
# LDA + downstream tasks
# ------------------------------

def flatten_doc(sentences: List[List[int]]) -> List[int]:
    return [tok for s in sentences for tok in s]

def tokens_to_dtm(docs_tokens: List[List[int]], V: int) -> csr_matrix:
    """CSR doc-term matrix from list of token-id lists."""
    indptr, indices, data = [0], [], []
    for toks in docs_tokens:
        cnt = Counter(toks)
        if cnt:
            k, v = zip(*cnt.items())
            indices.extend(k)
            data.extend(v)
        indptr.append(len(indices))
    return csr_matrix((np.asarray(data, dtype=np.int64),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int32)),
                      shape=(len(docs_tokens), V), dtype=np.float32)

def unit_to_aug_tokens(u: Unit) -> List[int]:
    toks: List[int] = []
    for s in u.doc_ctx: toks.extend(s)
    for s in u.candidates: toks.extend(s)
    return toks

def fit_lda_and_transform(X_train: csr_matrix,
                          X_tr_for_features: csr_matrix,
                          X_test: csr_matrix,
                          K: int, alpha: float, beta: float, seed: int):
    lda = LatentDirichletAllocation(
        n_components=K,
        doc_topic_prior=alpha,
        topic_word_prior=beta,
        learning_method="batch",
        max_iter=100,
        random_state=seed,
        evaluate_every=0,
    )
    _ = lda.fit_transform(X_train)
    W_train = lda.transform(X_tr_for_features)
    W_test  = lda.transform(X_test)
    comp = lda.components_.astype(np.float64) + 1e-12  # (K,V)
    phi_hat = comp / comp.sum(axis=1, keepdims=True)
    return lda, phi_hat, W_train, W_test

def align_topics(phi_true: np.ndarray, phi_hat: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Align phi_hat to phi_true via Hungarian on JSD; return (perm, mean_jsd, mean_cos)."""
    K = phi_true.shape[0]
    D = np.zeros((K, K), dtype=np.float64)
    C = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        pi = phi_true[i] / (phi_true[i].sum() + 1e-12)
        for j in range(K):
            pj = phi_hat[j] / (phi_hat[j].sum() + 1e-12)
            m = 0.5*(pi + pj)
            def _kl(a,b):
                mask = a > 0
                return float((a[mask] * (np.log(a[mask]/(b[mask]+1e-20) + 1e-20))).sum())
            js = 0.5*_kl(pi, m) + 0.5*_kl(pj, m)
            D[i,j] = js / np.log(2.0)
            C[i,j] = float(np.dot(pi, pj) / (np.linalg.norm(pi)*np.linalg.norm(pj) + 1e-20))
    r, c = linear_sum_assignment(D)
    return c, float(D[r, c].mean()), float(C[r, c].mean())


def evaluate_selected_doclevel(selected_by_doc: Dict[int, List[Unit]],
                               doc_total_units: Dict[int, int],
                               phi: np.ndarray,
                               phi_hat: np.ndarray,
                               lambda_obs: float,
                               rho: int) -> Dict[str, float]:
    """
    Evaluate a selection produced by a filtering rule that returns selected (accepted)
    augmented units per document.

    Parameters
    ----------
    selected_by_doc : dict {doc_idx -> list of accepted Units for that doc}
    doc_total_units : dict {doc_idx -> total #augmented units available for that doc}
    phi : (K, V) topic-word matrix, used for drift proxy
    lambda_obs : BAD/GOOD cutoff at doc-level
    rho : allowed #bad accepts per doc

    Returns
    -------
    dict with keys:
      - miscoverage : fraction of docs with (#bad accepts > rho)
      - accept_rate : (#accepted across all docs) / (total augmented across all docs)
      - distinct_1  : distinct-1 over all accepted replacement tokens
      - jsd_drift   : mean JSD(θ_true, θ_aug) proxy across accepted docs
    """
    n_docs = len(doc_total_units)
    if n_docs == 0:
        return {"miscoverage": 0.0, "accept_rate": 0.0, "distinct_1": 0.0, "jsd_drift": 0.0}

    miscover_cnt = 0
    accepted_total = 0
    total_units = 0
    all_accepted_tokens: List[int] = []
    jsd_vals: List[float] = []
    l1_vals: List[float] = []
    l2_vals: List[float] = []

    for doc_idx, tot in doc_total_units.items():
        accepted = selected_by_doc.get(doc_idx, [])
        accepted_total += len(accepted)
        total_units    += int(tot)

        # miscoverage: too many bad accepts for this doc?
        bad_accepts = sum(1 for u in accepted if float(u.A_obs_doc) < float(lambda_obs))
        if bad_accepts > rho:
            miscover_cnt += 1

        # diversity + drift per doc (only if at least one accept)
        if accepted:
            # collect all accepted replacement tokens once per doc (avoid duplicating context)
            for u in accepted:
                for sent in u.candidates:
                    all_accepted_tokens.extend(sent)

            # drift: compare θ_true vs a proxy θ_aug inferred from (doc_ctx + all accepted replacements)
            u0 = accepted[0]
            orig_theta = u0.doc_theta / (u0.doc_theta.sum() + 1e-20)
            aug_tokens = [t for s in u0.doc_ctx for t in s]
            for u in accepted:
                for s in u.candidates:
                    aug_tokens.extend(s)
            aug_theta = expected_theta_from_tokens(aug_tokens, phi, prior=orig_theta)
            jsd_vals.append(js_divergence(orig_theta, aug_theta))
            ##### add l1 norm and l2 norm
            l2_vals.append(np.linalg.norm(orig_theta - aug_theta, ord=2))
            l1_vals.append(np.linalg.norm(orig_theta - aug_theta, ord=1))

    return {
        "miscoverage": miscover_cnt / max(1, n_docs),
        "accept_rate": accepted_total / max(1, total_units),
        "distinct_1": distinct_1(all_accepted_tokens),
        "jsd_drift": float(np.mean(jsd_vals)) if jsd_vals else 0.0,
        "l1_drift": float(np.mean(l1_vals)) if l1_vals else 0.0,
        "l2_drift": float(np.mean(l2_vals)) if l2_vals else 0.0,
        "l1_drift_topic":np.mean(np.linalg.norm( phi- phi_hat, ord=1)),
        "l2_drift_topic": np.mean(np.linalg.norm( phi- phi_hat, ord=2))
    }


def evaluate_lda_and_downstream(cfg: SynthConfig,
                                res: Dict[str, Any],
                                seed: int = 0) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    phi        = res["phi"]
    docs       = res["docs"]
    aug_units  = res["aug_units"]
    idx_train  = res["splits"]["idx_train"]
    idx_aug    = res["splits"]["idx_aug"]

    # Originals for feature extraction (constant across variants)
    orig_train_tokens = [flatten_doc(docs[i].sentences) for i in idx_train]
    orig_test_tokens  = [flatten_doc(docs[i].sentences) for i in idx_aug]
    X_orig_train = tokens_to_dtm(orig_train_tokens, cfg.V)
    X_orig_test  = tokens_to_dtm(orig_test_tokens,  cfg.V)

    # TRAIN augmented units
    aug_train_units = [u for u in aug_units if u.doc_idx in set(idx_train)]

    # Ensure predictions exist on TRAIN units
    if any(np.isnan(getattr(u, "A_hat", np.nan)) or np.isnan(getattr(u, "A_obs_doc", np.nan))
           for u in aug_train_units):
        predict_for_units_doclevel(aug_train_units, res["reg"], res["idf"], res["phi"], target="obs")

    # ---------- Optional: build a conditional TRAIN set ----------
    # Reuse calibration units from results; if missing, we can rebuild from aug_units
    calib_units = res.get("calib_units", [u for u in aug_units if u.doc_idx in set(res["splits"]["idx_calib"])])
    cc_train = fit_conditional_threshold_doclevel(
        calib_units=calib_units,
        test_units=aug_train_units,
        idf=res["idf"],
        phi=res["phi"],
        lambda_obs=cfg.lambda_obs,
        alpha_cp=cfg.alpha_cp,
        rho=cfg.rho,
        verbose=False,
    )
    _, _, selected_by_doc_train, selected_train_units = cc_train

    def unit_to_aug_tokens(u: Unit) -> List[int]:
        toks = []
        for s in u.doc_ctx: toks.extend(s)
        for s in u.candidates: toks.extend(s)
        return toks

    # Build training variants
    train_variants: Dict[str, List[List[int]]] = {
        "original":         [flatten_doc(docs[i].sentences) for i in idx_train],
        "aug_unfiltered":   [unit_to_aug_tokens(u) for u in aug_train_units],
        "aug_observed":     [unit_to_aug_tokens(u) for u in aug_train_units
                             if not np.isnan(u.A_obs_doc) and u.A_obs_doc >= cfg.lambda_obs],
        "aug_conditional":  [unit_to_aug_tokens(u) for u in selected_train_units],
    }
    # Safety: if a bucket is empty, fall back to original
    for name, toks in list(train_variants.items()):
        if len(toks) == 0:
            print(f"[WARN] Variant '{name}' produced 0 training docs; falling back to 'original'.")
            train_variants[name] = train_variants["original"]

    X_train_variants = {name: tokens_to_dtm(toks, cfg.V) for name, toks in train_variants.items()}

    # Fit LDA & evaluate topic accuracy
    lda_results: Dict[str, Dict[str, Any]] = {}
    for name, Xtr in X_train_variants.items():
        lda, phi_hat, Wtr, Wte = fit_lda_and_transform(
            X_train=Xtr,
            X_tr_for_features=X_orig_train,
            X_test=X_orig_test,
            K=cfg.K, alpha=cfg.alpha, beta=cfg.beta, seed=seed
        )
        perm, mean_jsd, mean_cos = align_topics(phi, phi_hat)
        lda_results[name] = {
            "lda": lda, "phi_hat": phi_hat, "perm": perm,
            "phi_mean_jsd": mean_jsd, "phi_mean_cos": mean_cos,
            "W_train": Wtr, "W_test": Wte,
        }

    # Synthetic outcomes from true θ (fixed across variants)
    rng_out = np.random.default_rng(seed + 123)
    #### Iterate the beta vectors for regression and classification a bunch of times
    # to ensure that the downstream tasks are not biased by a single random seed.
    for exp_down in range(10):
        beta_reg = rng_out.normal(0, 1, size=cfg.K)
        beta_clf = rng_out.normal(0, 1, size=cfg.K)
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

        Theta_train = np.vstack([docs[i].theta for i in idx_train])
        Theta_test  = np.vstack([docs[i].theta for i in idx_aug])

        y_reg_train = Theta_train @ beta_reg + rng_out.normal(0, 0.1, size=len(idx_train))
        y_reg_test  = Theta_test  @ beta_reg + rng_out.normal(0, 0.1, size=len(idx_aug))

        logit_train = Theta_train @ beta_clf + rng_out.normal(0, 0.5, size=len(idx_train))
        logit_test  = Theta_test  @ beta_clf + rng_out.normal(0, 0.5, size=len(idx_aug))
        y_clf_train = (sigmoid(logit_train) > 0.5).astype(int)
        y_clf_test  = (sigmoid(logit_test)  > 0.5).astype(int)

        downstream_metrics: Dict[str, Dict[str, float]] = {}
        for name, resv in lda_results.items():
            Wtr, Wte = resv["W_train"], resv["W_test"]

            reg_model = Ridge(alpha=1.0, random_state=seed)
            reg_model.fit(Wtr, y_reg_train)
            y_pred = reg_model.predict(Wte)
            mse = mean_squared_error(y_reg_test, y_pred)
            r2  = r2_score(y_reg_test, y_pred)

            clf_model = LogisticRegression(max_iter=1000, random_state=seed)
            clf_model.fit(Wtr, y_clf_train)
            y_proba = clf_model.predict_proba(Wte)[:, 1]
            y_pred_cls = (y_proba >= 0.5).astype(int)
            auc = roc_auc_score(y_clf_test, y_proba)
            acc = accuracy_score(y_clf_test, y_pred_cls)

            downstream_metrics[name + "_" + exp_down] = {
                "name": name
                "reg_mse": float(mse), "reg_r2": float(r2),
                "clf_auc": float(auc), "clf_acc": float(acc),
            }
    # Aggregate results by converting to a DataFrame
    downstream_metrics_df = pd.DataFrame.from_dict(downstream_metrics, orient='index')
    downstream_metrics = downstream_metrics_df.groupby(name).mean().to_dict(orient='index')
        

    return lda_results, downstream_metrics

import pandas as pd

def results_to_row(cfg: SynthConfig,
                   res: Dict[str, Any],
                   lda_results: Dict[str, Any],
                   downstream_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    row = {
        # --- config ---
        "V": cfg.V, "K": cfg.K, "beta": cfg.beta, "alpha": cfg.alpha,
        "n_docs": cfg.n_docs, "S": cfg.S, "L": cfg.L, "mask_frac": cfg.mask_frac,
        "Kgen": cfg.Kgen, "delta": cfg.delta, "epsilon": cfg.epsilon, "T": cfg.T,
        "lambda_obs": cfg.lambda_obs, "rho": cfg.rho, "alpha_cp": cfg.alpha_cp,
        "n_train_docs": cfg.n_train_docs, "n_calib_docs": cfg.n_calib_docs, "n_aug_docs": cfg.n_aug_docs,
        "seed": cfg.seed,
        # --- conditional selection metrics ---
        "cond_miscoverage": res["conditional"]["metrics"]["miscoverage"],
        "cond_accept_rate": res["conditional"]["metrics"]["accept_rate"],
        "cond_distinct_1":  res["conditional"]["metrics"]["distinct_1"],
        "cond_jsd_drift":   res["conditional"]["metrics"]["jsd_drift"],
        # --- baselines ---
        "base_unf_miscoverage": res["baselines"]["unfiltered"]["miscoverage"],
        "base_unf_accept_rate": res["baselines"]["unfiltered"]["accept_rate"],
        "base_obs_miscoverage": res["baselines"]["observed_filter"]["miscoverage"],
        "base_obs_accept_rate": res["baselines"]["observed_filter"]["accept_rate"],
    }

    # LDA/topic alignment + downstream (optional; add whichever buckets exist)
    for bucket in ["original", "aug_unfiltered", "aug_observed", "aug_conditional"]:
        if bucket in lda_results and bucket in downstream_metrics:
            row[f"topic_jsd_{bucket}"] = float(lda_results[bucket]["phi_mean_jsd"])
            row[f"topic_cos_{bucket}"] = float(lda_results[bucket]["phi_mean_cos"])
            row[f"reg_mse_{bucket}"]   = float(downstream_metrics[bucket]["reg_mse"])
            row[f"reg_r2_{bucket}"]    = float(downstream_metrics[bucket]["reg_r2"])
            row[f"clf_auc_{bucket}"]   = float(downstream_metrics[bucket]["clf_auc"])
            row[f"clf_acc_{bucket}"]   = float(downstream_metrics[bucket]["clf_acc"])
    return row


# ------------------------------
# Orchestration
# ------------------------------

def run_synthetic_experiment(cfg: SynthConfig,
                             target_for_reg: str = "observed",
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Full pipeline:
      1) Sample topics phi and corpus (theta, z, sentences)
      2) Build base units with Kgen candidates per masked sent (doc-level)
      3) Build Kgen augmented doc Units per original doc
      4) Train A_hat regressor on TRAIN augmented docs
      5) Calibrate global s* via split-conformal on CALIB docs
      6) Evaluate on AUG (CP filter vs baselines)
    """
    set_seed(cfg.seed)
    rng = default_rng(cfg.seed)

    # 1) topics & corpus
    phi  = make_topics(cfg, rng)
    docs = generate_corpus(cfg, phi, rng)

    # 2) split by doc index
    idx_all   = np.arange(cfg.n_docs)
    idx_train = idx_all[:cfg.n_train_docs]
    idx_calib = idx_all[cfg.n_train_docs: cfg.n_train_docs + cfg.n_calib_docs]
    idx_aug   = idx_all[cfg.n_train_docs + cfg.n_calib_docs:
                        cfg.n_train_docs + cfg.n_calib_docs + cfg.n_aug_docs]

    # 3) build base units and augmented units
    base_units, base_by_doc, all_sents = build_units(cfg, docs, phi, rng)
    aug_units, aug_docs = assemble_augmented_docs_and_units(docs, base_by_doc, cfg.Kgen)

    # 4) IDF
    idf = compute_idf(all_sents, cfg.V)

    # 5) Train regressor on TRAIN augmented docs
    train_units = [u for u in aug_units if u.doc_idx in set(idx_train)]
    X_train, y_train, _ = assemble_feature_label_arrays_doclevel(
        train_units, idf, phi, target=target_for_reg, oracle_mode="cosine", as_log=False
    )
    reg = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=cfg.seed, n_jobs=-1)
    reg.fit(X_train, y_train)

    import matplotlib.pyplot as plt

    _, y_train_oracle, _ = assemble_feature_label_arrays_doclevel(train_units,  idf, phi, target="oracle",
    oracle_mode="cosine",  # or "mean_prob" if you prefer
    as_log=False)
    

    calib_units = [u for u in aug_units if u.doc_idx in set(idx_calib)]
    X_calib, y_calib, idx_calib_triplets = assemble_feature_label_arrays_doclevel(calib_units, idf, phi, target="observed",
        oracle_mode="cosine",  # or "mean_prob" if you prefer
        as_log=False)

    _, y_calib_oracle, _ = assemble_feature_label_arrays_doclevel(calib_units, idf, phi, target="oracle",
        oracle_mode="cosine",  # or "mean_prob" if you prefer
        as_log=False)


    r2 = np.corrcoef(reg.predict(X_calib), y_calib)[0, 1]

    #### Make nice plot
    plt.figure(figsize=(10, 6))
    plt.scatter(reg.predict(X_train), y_train, alpha=0.3, label='Train Predictions')
    plt.scatter(reg.predict(X_calib), y_calib, alpha=0.3, label='Calib Predictions')
    plt.xlabel('Predicted A_hat')
    plt.ylabel('True A_obs')
    plt.title('Predicted vs True A_obs: correlation on calib = {:.3f}'.format(r2))
    plt.legend()
    plt.grid(True)
    plt.show()


    import matplotlib.pyplot as plt
    ### Make nice plot
    r2 = np.corrcoef(reg.predict(X_calib), y_calib_oracle)[0, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(reg.predict(X_train), y_train_oracle, alpha=0.3, label='Train Predictions')
    plt.scatter(reg.predict(X_calib), y_calib_oracle, alpha=0.3, label='Calib Predictions')
    plt.xlabel('Predicted A_hat')
    plt.ylabel('Oracle A_star')
    plt.title('Predicted vs Oracle A_star: correlation on calib = {:.3f}'.format(r2))
    plt.legend()
    plt.grid(True)
    plt.show()


    

    predict_for_units_doclevel(train_units, reg, idf, phi, target="obs")  # <<< NEW

    # Predict A_hat for CALIB and AUG docs (also sets A_obs_doc)
    calib_units = [u for u in aug_units if u.doc_idx in set(idx_calib)]
    eval_units  = [u for u in aug_units if u.doc_idx in set(idx_aug)]

    if verbose:
        print(f"Training on {len(train_units)} augmented docs; calibrating on {len(calib_units)}; evaluating on {len(eval_units)}.")

    predict_for_units_doclevel(calib_units, reg, idf, phi, target="obs")
    predict_for_units_doclevel(eval_units,  reg, idf, phi, target="obs")

    # Build doc->units dicts (same Unit objects)
    calib_by_doc: Dict[int, List[Unit]] = {}
    for u in calib_units: calib_by_doc.setdefault(u.doc_idx, []).append(u)
    aug_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:  aug_by_doc.setdefault(u.doc_idx, []).append(u)

    # 6) Global CP threshold & metrics

    cc_model, thresholds_by_doc, selected_by_doc, selected_units = fit_conditional_threshold_doclevel(
                        calib_units=calib_units,
                        test_units=eval_units,
                        idf=idf,
                        phi=phi,
                        lambda_obs=0.06,
                        alpha_cp=0.05,
                        rho = 0,
                        verbose=True,
                    )
    
    # Total augmented per eval doc (denominator for accept rate)
    doc_total_eval: Dict[int, int] = {}
    for u in eval_units:
        doc_total_eval[u.doc_idx] = doc_total_eval.get(u.doc_idx, 0) + 1

    # Conditional selection metrics
    cc_metrics = evaluate_selected_doclevel(selected_by_doc,
                                            doc_total_eval,
                                            phi,
                                            lambda_obs=cfg.lambda_obs,
                                            rho=cfg.rho)

    # Baseline: accept everything
    selected_unfiltered_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        selected_unfiltered_by_doc.setdefault(u.doc_idx, []).append(u)
    baseline_unfiltered = evaluate_selected_doclevel(selected_unfiltered_by_doc,
                                                     doc_total_eval, phi, cfg.lambda_obs, cfg.rho)

    # Baseline: accept if observed >= lambda
    selected_observed_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        if float(u.A_obs_doc) >= float(cfg.lambda_obs):
            selected_observed_by_doc.setdefault(u.doc_idx, []).append(u)
    baseline_observed = evaluate_selected_doclevel(selected_observed_by_doc,
                                                   doc_total_eval, phi, cfg.lambda_obs, cfg.rho)

    # Pretty print
    n_sel = sum(len(v) for v in selected_by_doc.values())
    n_tot = sum(doc_total_eval.values())
    if verbose:
        print(f"[CC-LDA] Selected generations: {n_sel} / {n_tot} "
              f"({100.0 * n_sel / max(1, n_tot):.1f}%)")
        print(f"[CC-LDA] Metrics (conditional): {cc_metrics}")
        print(f"[BASE] Unfiltered: {baseline_unfiltered}")
        print(f"[BASE] Observed   : {baseline_observed}")

    # Package results once (no s_global in conditional pipeline)
    results = {
        "config": cfg,
        "splits": {"idx_train": idx_train, "idx_calib": idx_calib, "idx_aug": idx_aug},
        "phi": phi,
        "docs": docs,
        "idf": idf,
        "reg": reg,
        "aug_units": aug_units,
        "train_units": train_units,
        "calib_units": calib_units,
        "eval_units": eval_units,
        "conditional": {
            "model": cc_model,
            "thresholds_by_doc": thresholds_by_doc,
            "selected_by_doc": selected_by_doc,
            "selected_units": selected_units,
            "metrics": cc_metrics,
        },
        "baselines": {
            "unfiltered": baseline_unfiltered,
            "observed_filter": baseline_observed,
        }
    }
    return results



# =========================
# Example run
# =========================
if __name__ == "__main__":
    import pandas as pd
    results_df = pd.DataFrame()
    # You set rho=10 below; for doc-level CP the natural setting is rho=0.

    for alpha_cp in [0.05, 0.1, 0.2]:
                        cfg = SynthConfig(
                            V=1000, K=3, beta=0.1, alpha=0.3,
                            n_docs=n_train + n_calib + 500, S=10, L=12, mask_frac=0.5, Kgen=20,
                            delta=delta, epsilon=epsilon, T=temp,
                            lambda_obs=0.01, rho=10, alpha_cp=alpha_cp,
                            n_train_docs=n_train, n_calib_docs=n_calib n_aug_docs=500,
                            seed=SEED
                        )
                        for lambda_obs in [0.05, 0.1, 0.25, 0.35]:
                            for rho in [0, 1, 2, 5, 10]:
                                cfg.rho = rho
                                cfg.alpha_cp = alpha_cp
                                cfg.lambda_obs = lambda_obs
                                res = run_synthetic_experiment(cfg)
                                print(f"Config: {cfg}, Results: {res['conditional']['metrics']}")
                                lda_results, downstream_metrics = evaluate_lda_and_downstream(cfg, res, seed=cfg.seed)
                                row = results_to_row(cfg, res, lda_results, downstream_metrics)
                                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                                results_df.to_csv("synthetic_results_llm_cp_" +name_exp + ".csv", index=False)

                                print(results_df.tail(1))  # last row just added
                                print("Done")



