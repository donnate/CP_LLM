# =========================
# Clean, doc-level pipeline (rewritten & fixed)
# =========================

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

import numpy as np
from numpy.random import default_rng

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

import pandas as pd

# ---- CLI args ----
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

SEED     = args.seed
name_exp = args.exp_name
n_train  = args.n_train
n_calib  = args.n_calib
temp     = args.temp
delta    = args.delta
epsilon  = args.epsilon

# ---- local modules (expected to exist in your repo) ----
from config import (
    SynthConfig, Document, Candidate,
    make_topics, generate_corpus
)
from unit import (
    Unit, assemble_augmented_docs_and_units, build_units,
    assemble_feature_label_arrays_doclevel
)
from helper_similarity_metrics import (
    distinct_1, js_divergence
)
from cp_selection import (
    per_doc_S_doclevel, per_doc_S_doclevel_multi, global_threshold_S_doclevel,
    fit_conditional_threshold_doclevel
)
from featurize_document import compute_idf


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


# ------------------------------
# Scores & features (doc-level)
# ------------------------------

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
# Infer θ from tokens (given φ)
# ------------------------------

def expected_theta_from_tokens(tokens: List[int],
                               phi: np.ndarray,
                               prior: Optional[np.ndarray] = None) -> np.ndarray:
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


# ------------------------------
# Token helpers
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
    """
    Build an augmented *document* as (context + generated sentences).
    Keep this consistent with LDA training and similarity computation.
    """
    toks: List[int] = []
    for s in u.doc_ctx: toks.extend(s)
    for s in u.candidates: toks.extend(s)
    return toks

def original_tokens_for(docs: List[Document], idxs: Optional[np.ndarray] = None) -> List[List[int]]:
    if idxs is None:
        idxs = range(len(docs))
    return [flatten_doc(docs[i].sentences) for i in idxs]

def units_to_aug_docs(units: List[Unit]) -> List[List[int]]:
    """Each Unit becomes one 'augmented doc' (context + generated)."""
    return [unit_to_aug_tokens(u) for u in units]


# ------------------------------
# LDA fit + transforms
# ------------------------------

def fit_lda_and_transform_many(X_train: csr_matrix,
                               X_list_to_transform: List[csr_matrix],
                               K: int, alpha: float, beta: float, seed: int):
    """
    Fit an LDA on X_train, then .transform() every matrix in X_list_to_transform.
    Returns (lda, phi_hat, [W_0, W_1, ...]) where W_i = transform(X_list_to_transform[i]).
    """
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
    comps = lda.components_.astype(np.float64) + 1e-12
    phi_hat = comps / comps.sum(axis=1, keepdims=True)
    W_list = [lda.transform(X) for X in X_list_to_transform]
    return lda, phi_hat, W_list

def align_topics(phi_true: np.ndarray, phi_hat: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Align phi_hat to phi_true via Hungarian on JSD; return (perm, mean_jsd, mean_cos)."""
    K = phi_true.shape[0]
    D = np.zeros((K, K), dtype=np.float64)
    C = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        pi = phi_true[i] / (phi_true[i].sum() + 1e-12)
        for j in range(K):
            pj = phi_hat[j] / (phi_hat[j].sum() + 1e-12)
            m = 0.5 * (pi + pj)
            def _kl(a, b):
                mask = a > 0
                return float((a[mask] * (np.log(a[mask] / (b[mask] + 1e-20) + 1e-20))).sum())
            js = 0.5 * _kl(pi, m) + 0.5 * _kl(pj, m)
            D[i, j] = js / np.log(2.0)
            C[i, j] = float(np.dot(pi, pj) / (np.linalg.norm(pi) * np.linalg.norm(pj) + 1e-20))
    r, c = linear_sum_assignment(D)
    return c, float(D[r, c].mean()), float(C[r, c].mean())

def _align_and_doc_theta_metrics(phi_true: np.ndarray,
                                 phi_hat: np.ndarray,
                                 W_docs: np.ndarray,
                                 Theta_true: np.ndarray) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Align phi_hat to phi_true; reorder W_docs columns; compute topic + doc-theta metrics.
    """
    perm, mean_jsd, mean_cos = align_topics(phi_true, phi_hat)
    phi_hat_aligned = phi_hat[perm]
    W_aligned = W_docs[:, perm]
    l1 = float(np.mean(np.linalg.norm(W_aligned - Theta_true, ord=1, axis=1)))
    l2 = float(np.mean(np.linalg.norm(W_aligned - Theta_true, ord=2, axis=1)))
    return {
        "phi_mean_jsd": float(mean_jsd),
        "phi_mean_cos": float(mean_cos),
        "theta_l1_mean": l1,
        "theta_l2_mean": l2,
    }, phi_hat_aligned, W_aligned


# ------------------------------
# Diversity metrics
# ------------------------------

def _corpus_ngram_set(token_lists: List[List[int]], n: int = 1) -> set:
    S = set()
    if n == 1:
        for doc in token_lists:
            S.update(doc)
    else:
        for doc in token_lists:
            if len(doc) >= n:
                S.update(tuple(doc[i:i+n]) for i in range(len(doc)-n+1))
    return S

def _distinct_n(token_lists: List[List[int]], n: int = 1) -> float:
    total = sum(max(0, len(doc)-n+1) if n > 1 else len(doc) for doc in token_lists)
    if total == 0:
        return 0.0
    S = _corpus_ngram_set(token_lists, n=n)
    return len(S) / float(total)

def _lexical_diversity_metrics(orig_tokens: List[List[int]],
                               aug_tokens:  List[List[int]]) -> Dict[str, float]:
    d1_o = _distinct_n(orig_tokens, n=1)
    d2_o = _distinct_n(orig_tokens, n=2)
    d1_a = _distinct_n(orig_tokens + aug_tokens, n=1)
    d2_a = _distinct_n(orig_tokens + aug_tokens, n=2)

    Uo1 = _corpus_ngram_set(orig_tokens, n=1)
    Ua1 = _corpus_ngram_set(orig_tokens + aug_tokens, n=1)
    Uo2 = _corpus_ngram_set(orig_tokens, n=2)
    Ua2 = _corpus_ngram_set(orig_tokens + aug_tokens, n=2)

    new_vocab_frac = 0.0 if len(Ua1) == 0 else len(Ua1 - Uo1) / float(len(Ua1))
    jacc1 = 0.0 if (len(Uo1|Ua1) == 0) else len(Uo1 & Ua1) / float(len(Uo1 | Ua1))
    jacc2 = 0.0 if (len(Uo2|Ua2) == 0) else len(Uo2 & Ua2) / float(len(Uo2 | Ua2))

    return {
        "distinct1_unaug": d1_o, "distinct2_unaug": d2_o,
        "distinct1_with_aug": d1_a, "distinct2_with_aug": d2_a,
        "new_vocab_frac": new_vocab_frac,
        "jaccard_unigram": jacc1, "jaccard_bigram": jacc2,
    }

def _row_l2_normalize_dense(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def _effective_num_points_from_dense_unit_rows(Z: np.ndarray, block: int = 2048) -> float:
    """
    ENP = n^2 / sum_{ij} (cos_ij)^2 with L2-normalized rows.
    """
    n = Z.shape[0]
    if n == 0:
        return 0.0
    fro2 = 0.0
    for i in range(0, n, block):
        Gi = Z[i:i+block] @ Z.T
        fro2 += float(np.sum(Gi * Gi))
    return (n * n) / max(fro2, 1e-12)

def _dup_rate_from_dense_unit_rows(Z: np.ndarray, tau: float = 0.95, block: int = 2048) -> float:
    """
    Fraction of rows whose maximum cosine similarity to any other row >= tau.
    """
    n = Z.shape[0]
    if n <= 1:
        return 0.0
    hits = 0
    for i in range(0, n, block):
        b = min(block, n - i)
        Gi = Z[i:i+b] @ Z.T
        for r, j in enumerate(range(i, i+b)):
            Gi[r, j] = -1.0  # mask self
        hits += int(np.sum(np.max(Gi, axis=1) >= tau))
    return hits / float(n)

def _random_projection_enp_and_dups(X_csr: csr_matrix,
                                    rp_dim: int = 256,
                                    seed: int = 0,
                                    tau_dup: float = 0.95,
                                    block: int = 2048) -> Tuple[float, float]:
    """
    Projects TF csr matrix to low-dim dense space; returns (ENP, dup_rate).
    """
    n, V = X_csr.shape
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 1.0 / np.sqrt(rp_dim), size=(V, rp_dim))
    row_norms = np.sqrt(X_csr.multiply(X_csr).sum(axis=1)).A1 + 1e-12
    Xn = X_csr.multiply(1.0 / row_norms[:, None])  # csr, L2 row-norm 1
    Z = Xn @ R
    Z = _row_l2_normalize_dense(Z)
    enp = _effective_num_points_from_dense_unit_rows(Z, block=block)
    dup = _dup_rate_from_dense_unit_rows(Z, tau=tau_dup, block=block)
    return enp, dup

def _effective_num_points_topic(W: np.ndarray, block: int = 8192) -> float:
    if W.size == 0:
        return 0.0
    Z = _row_l2_normalize_dense(W)
    return _effective_num_points_from_dense_unit_rows(Z, block=block)

def _dup_rate_topic(W: np.ndarray, tau: float = 0.95, block: int = 8192) -> float:
    if W.size == 0:
        return 0.0
    Z = _row_l2_normalize_dense(W)
    return _dup_rate_from_dense_unit_rows(Z, tau=tau, block=block)


# ------------------------------
# Selection metrics (fast)
# ------------------------------

def evaluate_selected_doclevel(selected_by_doc: Dict[int, List[Unit]],
                               doc_total_units: Dict[int, int],
                               phi: np.ndarray,
                               lambda_obs: float,
                               rho: int,
                               *,
                               full_corpus_lda: bool = False,     # keep fast in selection eval
                               docs: Optional[List[Document]] = None,
                               splits: Optional[Dict[str, np.ndarray]] = None,
                               extra_selected_units: Optional[List[Unit]] = None,
                               seed: int = 0) -> Dict[str, Any]:
    """
    Compute selection metrics:
      miscoverage: fraction of original docs with > rho bad accepts
      accept_rate: (#accepted units) / (#total candidate units)
      distinct_1:  distinct-1 over accepted replacements
      jsd_drift:   JSD(θ_true, θ_aug) using φ as oracle to infer θ_aug
    """
    n_docs = len(doc_total_units)
    if n_docs == 0:
        return {"miscoverage": 0.0, "accept_rate": 0.0, "distinct_1": 0.0,
                "jsd_drift": 0.0, "l1_drift": 0.0, "l2_drift": 0.0}

    miscover_cnt, accepted_total, total_units = 0, 0, 0
    all_accepted_tokens: List[int] = []
    jsd_vals: List[float] = []
    l1_vals: List[float] = []
    l2_vals: List[float] = []

    for doc_idx, tot in doc_total_units.items():
        accepted = selected_by_doc.get(doc_idx, [])
        accepted_total += len(accepted)
        total_units += int(tot)
        bad_accepts = sum(1 for u in accepted if float(u.A_obs_doc) < float(lambda_obs))
        if bad_accepts > rho:
            miscover_cnt += 1

        if accepted:
            # distinct-1 over generated sentences only
            for u in accepted:
                for sent in u.candidates:
                    all_accepted_tokens.extend(sent)

            # doc-level drift: compare θ(orig) vs θ(aug=ctx+all gens)
            u0 = accepted[0]
            orig_theta = u0.doc_theta / (u0.doc_theta.sum() + 1e-20)
            aug_tokens = [t for s in u0.doc_ctx for t in s]
            for u in accepted:
                for s in u.candidates:
                    aug_tokens.extend(s)

            aug_theta = expected_theta_from_tokens(aug_tokens, phi, prior=orig_theta)
            jsd_vals.append(js_divergence(orig_theta, aug_theta))
            l2_vals.append(np.linalg.norm(orig_theta - aug_theta, ord=2))
            l1_vals.append(np.linalg.norm(orig_theta - aug_theta, ord=1))

    base = {
        "miscoverage": miscover_cnt / max(1, n_docs),
        "accept_rate": accepted_total / max(1, total_units),
        "distinct_1": distinct_1(all_accepted_tokens),
        "jsd_drift": float(np.mean(jsd_vals)) if jsd_vals else 0.0,
        "l1_drift": float(np.mean(l1_vals)) if l1_vals else 0.0,
        "l2_drift": float(np.mean(l2_vals)) if l2_vals else 0.0,
    }
    return base


# ------------------------------
# Similarity helpers (aug vs original)
# ------------------------------

def _cosine_from_token_lists(a_tokens: List[int], b_tokens: List[int]) -> float:
    ca, cb = Counter(a_tokens), Counter(b_tokens)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(ca[k] * cb.get(k, 0) for k in ca.keys())
    return dot / (na * nb)

def _summarize_aug_similarities(units: List[Unit],
                                docs: List[Document]) -> Dict[str, float]:
    """
    Cosine(units_as_full_doc, original_full_doc) for each Unit.
    Full doc = (context + generated) for consistency with LDA.
    """
    if not units:
        return {"n_added": 0, "cosine_mean": 0.0, "cosine_median": 0.0,
                "cosine_p10": 0.0, "cosine_p90": 0.0}
    sims = []
    for u in units:
        aug = unit_to_aug_tokens(u)
        orig = [tok for s in docs[u.doc_idx].sentences for tok in s]
        sims.append(_cosine_from_token_lists(aug, orig))
    arr = np.asarray(sims, dtype=np.float64)
    return {
        "n_added": len(units),
        "cosine_mean": float(arr.mean()),
        "cosine_median": float(np.median(arr)),
        "cosine_p10": float(np.percentile(arr, 10)),
        "cosine_p90": float(np.percentile(arr, 90)),
    }


# ------------------------------
# Full-corpus LDA + downstream on unseen set
# ------------------------------

def evaluate_lda_and_downstream(cfg: SynthConfig,
                                res: Dict[str, Any],
                                seed: int = 0,
                                n_unseen_eval_docs: int = 1000,
                                include_train_aug: bool = False,
                                include_calib_aug: bool = False
                                ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    LDA on FULL corpora (TRAIN+CALIB+TEST) with five variants and
    downstream evaluation on a NEW unseen synthetic set.

    Buckets:
      - 'unaug_full'          : originals only
      - 'aug_full_cp'         : originals + CP-selected eval (plus optional train/calib CP)
      - 'aug_full_unfiltered' : originals + all eval augmentations
      - 'aug_full_obs'        : originals + eval augmentations with A_obs >= lambda
      - 'aug_full_marginal'   : originals + marginal-CP-selected eval augmentations
    """
    rng = np.random.default_rng(seed + 202)

    phi_true   = res["phi"]
    docs       = res["docs"]
    aug_units  = res["aug_units"]
    idx_train  = np.asarray(res["splits"]["idx_train"])
    idx_calib  = np.asarray(res["splits"]["idx_calib"])
    idx_aug    = np.asarray(res["splits"]["idx_aug"])
    idx_all    = np.r_[idx_train, idx_calib, idx_aug]
    V          = phi_true.shape[1]

    # originals
    orig_all_tokens = [flatten_doc(docs[i].sentences) for i in idx_all]
    X_all_orig = tokens_to_dtm(orig_all_tokens, V)
    X_unaug_train = X_all_orig  # for 'unaug_full'

    # augmentation pools (EVAL)
    eval_units_all = [u for u in aug_units if u.doc_idx in set(idx_aug)]
    eval_units_obs = [u for u in eval_units_all
                      if not np.isnan(u.A_obs_doc) and u.A_obs_doc >= float(cfg.lambda_obs)]

    # CP-selected (EVAL)
    sel_by_doc_eval = res["conditional"]["selected_by_doc"]
    eval_units_cp = [u for lst in sel_by_doc_eval.values() for u in lst]

    # (Optional) add TRAIN/CALIB CP-selected to CP bucket
    extra_cp_units: List[Unit] = []
    if include_train_aug or include_calib_aug:
        train_units_all = [u for u in aug_units if u.doc_idx in set(idx_train)]
        calib_units_all = [u for u in aug_units if u.doc_idx in set(idx_calib)]
        if any(np.isnan(getattr(u, "A_hat", np.nan)) or np.isnan(getattr(u, "A_obs_doc", np.nan))
               for u in (train_units_all + calib_units_all)):
            predict_for_units_doclevel(train_units_all + calib_units_all,
                                       res["reg"], res["idf"], res["phi"], target="obs")

        # if include_train_aug:
        #     cc_train = fit_conditional_threshold_doclevel(
        #         calib_units=calib_units_all, test_units=train_units_all,
        #         idf=res["idf"], phi=res["phi"],
        #         lambda_obs=cfg.lambda_obs, alpha_cp=cfg.alpha_cp, rho=cfg.rho, verbose=False
        #     )
        #     _, _, _, sel_train_units = cc_train
        #     extra_cp_units.extend(sel_train_units)

        # if include_calib_aug:
        #     cc_calib = fit_conditional_threshold_doclevel(
        #         calib_units=calib_units_all, test_units=calib_units_all,
        #         idf=res["idf"], phi=res["phi"],
        #         lambda_obs=cfg.lambda_obs, alpha_cp=cfg.alpha_cp, rho=cfg.rho, verbose=False
        #     )
        #     _, _, _, sel_calib_units = cc_calib
        #     extra_cp_units.extend(sel_calib_units)

    # Marginal-CP-selected (EVAL)
    sel_by_doc_eval_marg = res.get("marginal_cp", {}).get("selected_by_doc", {})
    eval_units_marg = [u for lst in sel_by_doc_eval_marg.values() for u in lst]

    # Similarity stats for buckets
    simstats_by_bucket = {
        "unaug_full":          {"n_added": 0, "cosine_mean": 0.0, "cosine_median": 0.0,
                                "cosine_p10": 0.0, "cosine_p90": 0.0},
        "aug_full_cp":         _summarize_aug_similarities(eval_units_cp, docs),
        "aug_full_unfiltered": _summarize_aug_similarities(eval_units_all, docs),
        "aug_full_obs":        _summarize_aug_similarities(eval_units_obs, docs),
        "aug_full_marginal":   _summarize_aug_similarities(eval_units_marg, docs),
    }

    # training corpora for each bucket
    def units_to_docs(units: List[Unit]) -> List[List[int]]:
        return [unit_to_aug_tokens(u) for u in units]

    corpora_train = {
        "unaug_full":          orig_all_tokens,
        "aug_full_cp":         orig_all_tokens + units_to_docs(eval_units_cp),
        "aug_full_unfiltered": orig_all_tokens + units_to_docs(eval_units_all),
        "aug_full_obs":        orig_all_tokens + units_to_docs(eval_units_obs),
        "aug_full_marginal":   orig_all_tokens + units_to_docs(eval_units_marg),
    }
    X_train_by_bucket = {k: tokens_to_dtm(v, V) for k, v in corpora_train.items()}

    # new unseen set
    cfg_unseen = SynthConfig(
        V=cfg.V, K=cfg.K, beta=cfg.beta, alpha=cfg.alpha,
        n_docs=n_unseen_eval_docs, S=cfg.S, L=cfg.L,
        mask_frac=cfg.mask_frac, Kgen=cfg.Kgen,
        delta=cfg.delta, epsilon=cfg.epsilon, T=cfg.T,
        lambda_obs=cfg.lambda_obs, rho=cfg.rho, alpha_cp=cfg.alpha_cp,
        n_train_docs=0, n_calib_docs=0, n_aug_docs=0, seed=seed + 777
    )
    unseen_docs = generate_corpus(cfg_unseen, phi_true, rng)
    X_unseen_orig = tokens_to_dtm([flatten_doc(d.sentences) for d in unseen_docs], V)

    Theta_all_true = np.vstack([docs[i].theta for i in idx_all])
    Theta_unseen   = np.vstack([d.theta for d in unseen_docs])

    # fit one LDA per bucket; compute topic/theta recovery
    lda_results: Dict[str, Dict[str, float]] = {}
    W_all_aligned_by_bucket: Dict[str, np.ndarray]    = {}
    W_unseen_aligned_by_bucket: Dict[str, np.ndarray] = {}

    for bucket, Xtr in X_train_by_bucket.items():
        lda, phi_hat, [W_all, W_unseen] = fit_lda_and_transform_many(
            X_train=Xtr,
            X_list_to_transform=[X_all_orig, X_unseen_orig],
            K=cfg.K, alpha=cfg.alpha, beta=cfg.beta, seed=seed
        )
        met, phi_hat_aligned, W_all_aligned = _align_and_doc_theta_metrics(
            phi_true=phi_true, phi_hat=phi_hat, W_docs=W_all, Theta_true=Theta_all_true
        )

        # Add similarity counts/statistics
        ss = simstats_by_bucket.get(bucket)
        if ss is not None:
            met["n_aug_added"]          = int(ss["n_added"])
            met["aug2orig_cosine_mean"] = float(ss["cosine_mean"])
            met["aug2orig_cosine_med"]  = float(ss["cosine_median"])
            met["aug2orig_cosine_p10"]  = float(ss["cosine_p10"])
            met["aug2orig_cosine_p90"]  = float(ss["cosine_p90"])

        # align unseen as well
        perm, _, _ = align_topics(phi_true, phi_hat)
        W_unseen_aligned = W_unseen[:, perm]

        # topic drift norms (mean across topics)
        met["l1_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_hat_aligned, ord=1, axis=1)))
        met["l2_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_hat_aligned, ord=2, axis=1)))

        lda_results[bucket] = met
        W_all_aligned_by_bucket[bucket]     = W_all_aligned
        W_unseen_aligned_by_bucket[bucket]  = W_unseen_aligned

    # downstream on unseen; models trained on ALL originals (for each bucket's LDA space)
    results_runs = []
    for rep in range(10):
        beta_reg = rng.normal(0, 1, size=cfg.K)
        beta_clf = rng.normal(0, 1, size=cfg.K)
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

        # outcomes from true thetas
        y_reg_all    = Theta_all_true @ beta_reg + rng.normal(0, 0.1, size=Theta_all_true.shape[0])
        y_reg_unseen = Theta_unseen   @ beta_reg + rng.normal(0, 0.1, size=Theta_unseen.shape[0])

        logit_all    = Theta_all_true @ beta_clf + rng.normal(0, 0.5, size=Theta_all_true.shape[0])
        logit_unseen = Theta_unseen   @ beta_clf + rng.normal(0, 0.5, size=Theta_unseen.shape[0])
        y_clf_all    = (sigmoid(logit_all)    > 0.5).astype(int)
        y_clf_unseen = (sigmoid(logit_unseen) > 0.5).astype(int)

        for bucket in ["unaug_full", "aug_full_cp", "aug_full_unfiltered", "aug_full_obs", "aug_full_marginal"]:
            W_all    = W_all_aligned_by_bucket[bucket]
            W_unseen = W_unseen_aligned_by_bucket[bucket]

            # OLS regression
            ols = LinearRegression()
            ols.fit(W_all, y_reg_all)
            y_pred = ols.predict(W_unseen)
            mse = mean_squared_error(y_reg_unseen, y_pred)
            r2  = r2_score(y_reg_unseen, y_pred)

            # Logistic classification
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(W_all, y_clf_all)
            y_prob = clf.predict_proba(W_unseen)[:, 1]
            y_hat  = (y_prob >= 0.5).astype(int)
            auc = roc_auc_score(y_clf_unseen, y_prob)
            acc = accuracy_score(y_clf_unseen, y_hat)

            results_runs.append({
                "bucket": bucket,
                "reg_mse": float(mse), "reg_r2": float(r2),
                "clf_auc": float(auc), "clf_acc": float(acc),
            })

    df = pd.DataFrame(results_runs)
    downstream_metrics = df.groupby("bucket", sort=False).mean(numeric_only=True).to_dict(orient="index")
    return lda_results, downstream_metrics


# ------------------------------
# Result packaging
# ------------------------------

def results_to_row(cfg: SynthConfig,
                   res: Dict[str, Any],
                   lda_results: Dict[str, Any],
                   downstream_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    row = {
        # config
        "V": cfg.V, "K": cfg.K, "beta": cfg.beta, "alpha": cfg.alpha,
        "n_docs": cfg.n_docs, "S": cfg.S, "L": cfg.L, "mask_frac": cfg.mask_frac,
        "Kgen": cfg.Kgen, "delta": cfg.delta, "epsilon": cfg.epsilon, "T": cfg.T,
        "lambda_obs": cfg.lambda_obs, "rho": cfg.rho, "alpha_cp": cfg.alpha_cp,
        "n_train_docs": cfg.n_train_docs, "n_calib_docs": cfg.n_calib_docs, "n_aug_docs": cfg.n_aug_docs,
        "seed": cfg.seed,

        # monitor on calib (can be NaN if unavailable)
        "r2_calib":           res.get("r2_calib", np.nan),
        "r2_calib_oracle":    res.get("r2_calib_oracle", np.nan),
        "corr_calib":         res.get("corr_calib", np.nan),
        "corr_calib_oracle":  res.get("corr_calib_oracle", np.nan),

        # conditional CP selection metrics
        "CP_miscoverage": res["conditional"]["metrics"]["miscoverage"],
        "CP_accept_rate": res["conditional"]["metrics"]["accept_rate"],
        "CP_distinct_1":  res["conditional"]["metrics"]["distinct_1"],
        "CP_jsd_drift":   res["conditional"]["metrics"]["jsd_drift"],

        # baselines selection metrics
        "base_unf_miscoverage": res["baselines"]["unfiltered"]["miscoverage"],
        "base_unf_accept_rate": res["baselines"]["unfiltered"]["accept_rate"],
        "base_obs_miscoverage": res["baselines"]["observed_filter"]["miscoverage"],
        "base_obs_accept_rate": res["baselines"]["observed_filter"]["accept_rate"],
    }

    # marginal CP selection metrics
    row.update({
        "Marg_miscoverage": res.get("marginal_cp", {}).get("metrics", {}).get("miscoverage", np.nan),
        "Marg_accept_rate": res.get("marginal_cp", {}).get("metrics", {}).get("accept_rate", np.nan),
        "Marg_distinct_1":  res.get("marginal_cp", {}).get("metrics", {}).get("distinct_1", np.nan),
        "Marg_jsd_drift":   res.get("marginal_cp", {}).get("metrics", {}).get("jsd_drift", np.nan),
    })

    # similarity summaries (what we actually add during selection on eval)
    sim_cp  = res.get("conditional", {}).get("additions_summary", {})
    if sim_cp:
        row["n_added_cp"] = int(sim_cp["n_added"])
        row["cosine_mean_cp"] = float(sim_cp["cosine_mean"])
        row["cosine_med_cp"]  = float(sim_cp["cosine_median"])

    sim_unf = res.get("baselines", {}).get("unfiltered", {}).get("additions_summary", {})
    if sim_unf:
        row["n_added_unfiltered"] = int(sim_unf["n_added"])
        row["cosine_mean_unfiltered"] = float(sim_unf["cosine_mean"])
        row["cosine_med_unfiltered"]  = float(sim_unf["cosine_median"])

    sim_obs = res.get("baselines", {}).get("observed_filter", {}).get("additions_summary", {})
    if sim_obs:
        row["n_added_obs"] = int(sim_obs["n_added"])
        row["cosine_mean_obs"] = float(sim_obs["cosine_mean"])
        row["cosine_med_obs"]  = float(sim_obs["cosine_median"])

    sim_marg = res.get("marginal_cp", {}).get("additions_summary", {})
    if sim_marg:
        row["n_added_marg"] = int(sim_marg["n_added"])
        row["cosine_mean_marg"] = float(sim_marg["cosine_mean"])
        row["cosine_med_marg"]  = float(sim_marg["cosine_median"])

    # full-corpus LDA / downstream buckets (ensure consistent keys)
    desired_buckets = [
        ("unaug_full",          "unaug_full"),
        ("aug_full_cp",         "aug_full_cp"),
        ("aug_full_unfiltered", "aug_full_unfiltered"),
        ("aug_full_obs",        "aug_full_obs"),
        ("aug_full_marginal",   "aug_full_marginal"),
    ]

    # backward compat: if only "aug_full" exists, map it to "aug_full_cp"
    if ("aug_full_cp" not in lda_results and "aug_full_cp" not in downstream_metrics
        and ("aug_full" in lda_results or "aug_full" in downstream_metrics)):
        lda_results.setdefault("aug_full_cp", lda_results.get("aug_full", {}))
        downstream_metrics.setdefault("aug_full_cp", downstream_metrics.get("aug_full", {}))

    for key, tag in desired_buckets:
        if key in lda_results:
            lr = lda_results[key]
            if "phi_mean_jsd" in lr: row[f"topic_jsd_{tag}"] = float(lr["phi_mean_jsd"])
            if "phi_mean_cos" in lr: row[f"topic_cos_{tag}"] = float(lr["phi_mean_cos"])
            if "theta_l1_mean" in lr: row[f"theta_l1_{tag}"] = float(lr["theta_l1_mean"])
            if "theta_l2_mean" in lr: row[f"theta_l2_{tag}"] = float(lr["theta_l2_mean"])
            if "l1_drift_topic" in lr: row[f"phi_l1_{tag}"] = float(lr["l1_drift_topic"])
            if "l2_drift_topic" in lr: row[f"phi_l2_{tag}"] = float(lr["l2_drift_topic"])
            if "n_aug_added" in lr:           row[f"n_added_{tag}"] = int(lr["n_aug_added"])
            if "aug2orig_cosine_mean" in lr:  row[f"cosine_mean_{tag}"] = float(lr["aug2orig_cosine_mean"])
            if "aug2orig_cosine_med" in lr:   row[f"cosine_med_{tag}"]  = float(lr["aug2orig_cosine_med"])

        if key in downstream_metrics:
            dm = downstream_metrics[key]
            if "reg_mse" in dm: row[f"reg_mse_{tag}"] = float(dm["reg_mse"])
            if "reg_r2"  in dm: row[f"reg_r2_{tag}"]  = float(dm["reg_r2"])
            if "clf_auc" in dm: row[f"clf_auc_{tag}"] = float(dm["clf_auc"])
            if "clf_acc" in dm: row[f"clf_acc_{tag}"] = float(dm["clf_acc"])

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
      2) Build base/augmented Units
      3) Train A_hat regressor on TRAIN Units
      4) Fit conditional thresholds on CALIB; select on EVAL
      5) Evaluate selection metrics and record similarity summaries
      6) Compute marginal CP as a baseline (global threshold)
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

    # Monitoring on CALIB
    calib_units = [u for u in aug_units if u.doc_idx in set(idx_calib)]
    X_calib, y_calib, _ = assemble_feature_label_arrays_doclevel(
        calib_units, idf, phi, target="observed", oracle_mode="cosine", as_log=False
    )
    _, y_calib_oracle, _ = assemble_feature_label_arrays_doclevel(
        calib_units, idf, phi, target="oracle", oracle_mode="cosine", as_log=False
    )
    pred_c = reg.predict(X_calib)
    corr_calib_ahat   = float(np.corrcoef(pred_c, y_calib)[0, 1]) if X_calib.shape[0] > 1 else np.nan
    corr_calib_oracle = float(np.corrcoef(pred_c, y_calib_oracle)[0, 1]) if X_calib.shape[0] > 1 else np.nan
    r2_calib          = r2_score(y_calib, pred_c) if X_calib.shape[0] > 1 else np.nan
    r2_calib_oracle   = r2_score(y_calib_oracle, pred_c) if X_calib.shape[0] > 1 else np.nan
    if verbose:
        print(f"[MON] corr(pred, A_obs) calib={corr_calib_ahat:.3f} | corr(pred, A_star) calib={corr_calib_oracle:.3f}")

    # Predict A_hat for TRAIN/CALIB/EVAL
    eval_units  = [u for u in aug_units if u.doc_idx in set(idx_aug)]
    predict_for_units_doclevel(train_units, reg, idf, phi, target="obs")
    predict_for_units_doclevel(calib_units, reg, idf, phi, target="obs")
    predict_for_units_doclevel(eval_units,  reg, idf, phi, target="obs")

    # 6) Conditional thresholds (fit on calib, apply to eval)
    if verbose:
        print(f"Training on {len(train_units)} augmented docs; calibrating on {len(calib_units)}; evaluating on {len(eval_units)}.")

    cc_model, thresholds_by_doc, selected_by_doc, selected_units = fit_conditional_threshold_doclevel(
        calib_units=calib_units,
        test_units=eval_units,
        idf=idf, phi=phi,
        lambda_obs=cfg.lambda_obs,
        alpha_cp=cfg.alpha_cp,
        rho=cfg.rho,
        verbose=True,
    )

    # per-doc denominators for accept-rate on eval
    doc_total_eval: Dict[int, int] = {}
    for u in eval_units:
        doc_total_eval[u.doc_idx] = doc_total_eval.get(u.doc_idx, 0) + 1

    # Selection metrics (fast; no LDA here)
    cc_metrics = evaluate_selected_doclevel(
        selected_by_doc, doc_total_eval, phi, lambda_obs=cfg.lambda_obs, rho=cfg.rho,
        full_corpus_lda=False
    )

    # Baselines on eval
    selected_unfiltered_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        selected_unfiltered_by_doc.setdefault(u.doc_idx, []).append(u)
    baseline_unfiltered = evaluate_selected_doclevel(
        selected_unfiltered_by_doc, doc_total_eval, phi, cfg.lambda_obs, cfg.rho, full_corpus_lda=False
    )

    selected_observed_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        if not np.isnan(u.A_obs_doc) and float(u.A_obs_doc) >= float(cfg.lambda_obs):
            selected_observed_by_doc.setdefault(u.doc_idx, []).append(u)
    baseline_observed = evaluate_selected_doclevel(
        selected_observed_by_doc, doc_total_eval, phi, cfg.lambda_obs, cfg.rho, full_corpus_lda=False
    )

    # Marginal CP (global threshold): build calib_by_doc dict first
    calib_by_doc: Dict[int, List[Unit]] = {}
    for u in calib_units:
        calib_by_doc.setdefault(u.doc_idx, []).append(u)

    s_global_marginal = global_threshold_S_doclevel(
        calib_by_doc, lambda_obs=cfg.lambda_obs, rho=cfg.rho, alpha_cp=cfg.alpha_cp
    )

    selected_marginal_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        if float(u.A_hat) >= float(s_global_marginal):
            selected_marginal_by_doc.setdefault(u.doc_idx, []).append(u)

    marginal_metrics = evaluate_selected_doclevel(
        selected_marginal_by_doc, doc_total_eval, phi, lambda_obs=cfg.lambda_obs, rho=cfg.rho,
        full_corpus_lda=False
    )

    # additions / similarity summaries for what we add on EVAL
    eval_units_cp   = [u for lst in selected_by_doc.values() for u in lst]
    eval_units_all  = eval_units[:]  # all generated for eval docs
    eval_units_obs  = [u for u in eval_units if not np.isnan(u.A_obs_doc) and u.A_obs_doc >= float(cfg.lambda_obs)]
    eval_units_marg = [u for lst in selected_marginal_by_doc.values() for u in lst]

    sim_cp   = _summarize_aug_similarities(eval_units_cp, docs)
    sim_unf  = _summarize_aug_similarities(eval_units_all, docs)
    sim_obs  = _summarize_aug_similarities(eval_units_obs, docs)
    sim_marg = _summarize_aug_similarities(eval_units_marg, docs)

    if verbose:
        n_sel = sum(len(v) for v in selected_by_doc.values())
        n_tot = sum(doc_total_eval.values())
        print(f"[CC] Selected generations: {n_sel} / {n_tot} ({100.0 * n_sel / max(1, n_tot):.1f}%)")
        print(f"[CC] Metrics (conditional): {cc_metrics}")
        print(f"[BASE] Unfiltered: {baseline_unfiltered}")
        print(f"[BASE] Observed  : {baseline_observed}")
        print(f"[MARG] Metrics   : {marginal_metrics}")

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
        # monitors
        "corr_calib": corr_calib_ahat,
        "corr_calib_oracle": corr_calib_oracle,
        "r2_calib": r2_calib,
        "r2_calib_oracle": r2_calib_oracle,
        # selections
        "conditional": {
            "model": cc_model,
            "thresholds_by_doc": thresholds_by_doc,
            "selected_by_doc": selected_by_doc,
            "selected_units": selected_units,
            "metrics": cc_metrics,
            "additions_summary": sim_cp,
        },
        "marginal_cp": {
            "threshold": float(s_global_marginal),
            "selected_by_doc": selected_marginal_by_doc,
            "metrics": marginal_metrics,
            "additions_summary": sim_marg,
        },
        "baselines": {
            "unfiltered":     {**baseline_unfiltered, "additions_summary": sim_unf},
            "observed_filter": {**baseline_observed,  "additions_summary": sim_obs},
        }
    }
    return results


# =========================
# Example run
# =========================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame()

    for alpha_cp in [0.05, 0.1, 0.2]:
        cfg = SynthConfig(
            V=1000, K=3, beta=0.1, alpha=0.3,
            n_docs=n_train + n_calib + 500, S=10, L=12, mask_frac=0.5, Kgen=20,
            delta=delta, epsilon=epsilon, T=temp,
            lambda_obs=0.01, rho=10, alpha_cp=alpha_cp,
            n_train_docs=n_train, n_calib_docs=n_calib, n_aug_docs=500,
            seed=SEED
        )
        for lambda_obs in [0.05, 0.1, 0.25, 0.35]:
            for rho in [0, 1, 2, 5, 10]:
                cfg.rho = rho
                cfg.alpha_cp = alpha_cp
                cfg.lambda_obs = lambda_obs

                # Run full synthetic pipeline
                res = run_synthetic_experiment(cfg)
                print(f"Config: {cfg}, Results: {res['conditional']['metrics']}")

                # Full-corpus LDA + downstream on unseen set
                lda_results, downstream_metrics = evaluate_lda_and_downstream(cfg, res, seed=cfg.seed)

                # Single-row summary for this (cfg, seed)
                row = results_to_row(cfg, res, lda_results, downstream_metrics)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                out_path = os.path.join("results", f"synthetic_results_llm_cp_{name_exp}.csv")
                results_df.to_csv(out_path, index=False)

                print(results_df.tail(1))  # last row just added
                print("Done")
