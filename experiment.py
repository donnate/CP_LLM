# =========================
# Clean, doc-level pipeline
# =========================

from __future__ import annotations

import math, random, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence, Optional
from collections import Counter

import numpy as np
from numpy.random import default_rng

from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

import pandas as pd


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


# ---- local modules you already split out ----
from config import (
    SynthConfig, Document, Candidate,
    make_topics, generate_corpus, softmax_temp, dirichlet_sample
)
from unit import (
    BaseUnit, Unit, assemble_augmented_docs_and_units, gen_candidates_for_mask,
    build_units, assemble_feature_label_arrays_doclevel
)
from helper_similarity_metrics import (
    cosine_from_counts, cosine_from_freq, counts_vec, distinct_1, js_divergence
)
from cp_selection import (
    per_doc_S_doclevel, per_doc_S_doclevel_multi, global_threshold_S_doclevel,
    fit_conditional_threshold_doclevel
)
from featurize_document import featurize_candidate_doclevel, compute_idf


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
    all_accepted_tokens: List[int] = []
    jsd_vals: List[float] = []

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


# --- helpers for "full-corpus" LDA evaluation ---

def original_tokens_for(docs, idxs=None):
    if idxs is None:
        idxs = range(len(docs))
    return [flatten_doc(docs[i].sentences) for i in idxs]

def units_to_aug_docs(units: List[Unit]) -> List[List[int]]:
    # Each Unit is treated as its own doc for the augmented corpus
    return [unit_to_aug_tokens(u) for u in units]

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
            m = 0.5*(pi + pj)
            def _kl(a,b):
                mask = a > 0
                return float((a[mask] * (np.log(a[mask]/(b[mask]+1e-20) + 1e-20))).sum())
            js = 0.5*_kl(pi, m) + 0.5*_kl(pj, m)
            D[i,j] = js / np.log(2.0)
            C[i,j] = float(np.dot(pi, pj) / (np.linalg.norm(pi)*np.linalg.norm(pj) + 1e-20))
    r, c = linear_sum_assignment(D)
    return c, float(D[r, c].mean()), float(C[r, c].mean())

def _align_and_doc_theta_metrics(phi_true: np.ndarray,
                                 phi_hat: np.ndarray,
                                 W_docs: np.ndarray,  # shape (n_docs, K) from lda.transform
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
# Selection metrics + optional full-corpus LDA comparison
# ------------------------------

def evaluate_selected_doclevel(selected_by_doc: Dict[int, List[Unit]],
                               doc_total_units: Dict[int, int],
                               phi: np.ndarray,
                               lambda_obs: float,
                               rho: int,
                               *,
                               full_corpus_lda: bool = False,
                               docs: Optional[List[Document]] = None,
                               splits: Optional[Dict[str, np.ndarray]] = None,
                               extra_selected_units: Optional[List[Unit]] = None,
                               seed: int = 0) -> Dict[str, Any]:
    """
    (A) Original selection metrics (unchanged): miscoverage, accept_rate, diversity, doc-level drift.
    (B) OPTIONAL (full_corpus_lda=True): re-fit LDA on the entire corpus and compare topic/theta recovery
        - 'unaug_full' : LDA on original TRAIN+CALIB+TEST docs
        - 'aug_full'   : LDA on (unaug_full + selected augmented docs)

    Pass `extra_selected_units` to include accepted units from TRAIN/CALIB as well.
    """
    # ---------- (A) selection metrics ----------
    n_docs = len(doc_total_units)
    if n_docs == 0:
        base = {"miscoverage": 0.0, "accept_rate": 0.0, "distinct_1": 0.0,
                "jsd_drift": 0.0, "l1_drift": 0.0, "l2_drift": 0.0,
                "l1_drift_topic": 0.0, "l2_drift_topic": 0.0}
    else:
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
                for u in accepted:
                    for sent in u.candidates:
                        all_accepted_tokens.extend(sent)

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
            "l1_drift_topic": 0.0,   # filled only when full_corpus_lda=True
            "l2_drift_topic": 0.0,
        }

    if not full_corpus_lda:
        return base

    # ---------- (B) re-fit LDA on FULL CORPUS (with/without augmentation) ----------
    assert docs is not None and splits is not None, \
        "When full_corpus_lda=True, pass docs= and splits={'idx_train','idx_calib','idx_aug'}."

    V = phi.shape[1]
    idx_all = np.r_[splits["idx_train"], splits["idx_calib"], splits["idx_aug"]]

    # Unaugmented full corpus = all original docs
    orig_all_tokens = original_tokens_for(docs, idx_all)
    X_all_orig = tokens_to_dtm(orig_all_tokens, V)

    # Augmented docs = accepted units from selected_by_doc (+ optionally extra_selected_units)
    accepted_eval_units = [u for lst in selected_by_doc.values() for u in lst]
    extra = extra_selected_units or []
    aug_tokens = units_to_aug_docs(accepted_eval_units + extra)

    # Build train corpora for LDA
    X_unaug_train = tokens_to_dtm(orig_all_tokens, V)
    X_aug_train   = tokens_to_dtm(orig_all_tokens + aug_tokens, V)

    # Fit LDA on unaugmented full corpus; transform original docs
    lda_u, phi_u, [W_all_u] = fit_lda_and_transform_many(
        X_train=X_unaug_train,
        X_list_to_transform=[X_all_orig],
        K=phi.shape[0], alpha=1.0/phi.shape[0], beta=1.0/V, seed=seed
    )

    Theta_all_true = np.vstack([docs[i].theta for i in idx_all])
    metrics_u, phi_u_aligned, W_all_u_aligned = _align_and_doc_theta_metrics(
        phi_true=phi, phi_hat=phi_u, W_docs=W_all_u, Theta_true=Theta_all_true
    )

    # Fit LDA on augmented full corpus; transform original docs
    lda_a, phi_a, [W_all_a] = fit_lda_and_transform_many(
        X_train=X_aug_train,
        X_list_to_transform=[X_all_orig],
        K=phi.shape[0], alpha=1.0/phi.shape[0], beta=1.0/V, seed=seed
    )
    metrics_a, phi_a_aligned, W_all_a_aligned = _align_and_doc_theta_metrics(
        phi_true=phi, phi_hat=phi_a, W_docs=W_all_a, Theta_true=Theta_all_true
    )

    # Topic drift (phi vs phi_hat) as mean per-topic norms
    metrics_u["l1_drift_topic"] = float(np.mean(np.linalg.norm(phi - phi_u_aligned, ord=1, axis=1)))
    metrics_u["l2_drift_topic"] = float(np.mean(np.linalg.norm(phi - phi_u_aligned, ord=2, axis=1)))
    metrics_a["l1_drift_topic"] = float(np.mean(np.linalg.norm(phi - phi_a_aligned, ord=1, axis=1)))
    metrics_a["l2_drift_topic"] = float(np.mean(np.linalg.norm(phi - phi_a_aligned, ord=2, axis=1)))

    return {
        **base,
        "lda_full": {
            "unaug_full": metrics_u,
            "aug_full":   metrics_a,
        }
    }


# ------------------------------
# Full-corpus LDA + downstream on unseen set
# ------------------------------

def evaluate_lda_and_downstream(cfg: SynthConfig,
                                res: Dict[str, Any],
                                seed: int = 0,
                                n_unseen_eval_docs: int = 1000) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    LDA on FULL corpora (TRAIN+CALIB+TEST) with/without augmentation and
    downstream evaluation on a NEW unseen synthetic set.
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

    # Ensure predictions on train+calib units (needed for conditional selection)
    aug_train_units  = [u for u in aug_units if u.doc_idx in set(idx_train)]
    aug_calib_units  = [u for u in aug_units if u.doc_idx in set(idx_calib)]
    if any(np.isnan(getattr(u, "A_hat", np.nan)) or np.isnan(getattr(u, "A_obs_doc", np.nan))
           for u in (aug_train_units + aug_calib_units)):
        predict_for_units_doclevel(aug_train_units + aug_calib_units,
                                   res["reg"], res["idf"], res["phi"], target="obs")

    # Conditional selections for TRAIN and CALIB
    cc_train = fit_conditional_threshold_doclevel(
        calib_units=aug_calib_units, test_units=aug_train_units,
        idf=res["idf"], phi=res["phi"],
        lambda_obs=cfg.lambda_obs, alpha_cp=cfg.alpha_cp, rho=cfg.rho, verbose=False
    )
    _, _, sel_by_doc_train, sel_train_units = cc_train

    cc_calib = fit_conditional_threshold_doclevel(
        calib_units=aug_calib_units, test_units=aug_calib_units,
        idf=res["idf"], phi=res["phi"],
        lambda_obs=cfg.lambda_obs, alpha_cp=cfg.alpha_cp, rho=cfg.rho, verbose=False
    )
    _, _, sel_by_doc_calib, sel_calib_units = cc_calib

    # EVAL selections already available
    sel_by_doc_eval = res["conditional"]["selected_by_doc"]
    sel_eval_units  = [u for lst in sel_by_doc_eval.values() for u in lst]

    # Build corpora
    orig_all_tokens = original_tokens_for(docs, idx_all)                 # unaug_full originals
    all_selected_units = sel_train_units + sel_calib_units + sel_eval_units
    aug_tokens = units_to_aug_docs(all_selected_units)                   # accepted augmented docs

    X_unaug_train = tokens_to_dtm(orig_all_tokens, V)                    # train LDA on originals
    X_aug_train   = tokens_to_dtm(orig_all_tokens + aug_tokens, V)       # train LDA on originals + augmentations

    X_all_orig    = tokens_to_dtm(orig_all_tokens, V)                    # transform: originals only

    # NEW unseen test set (same phi_true)
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

    # Fit two LDAs and compute topic/theta recovery on ALL original docs; also transform unseen
    # Unaugmented full
    lda_u, phi_u, [W_all_u, W_unseen_u] = fit_lda_and_transform_many(
        X_train=X_unaug_train,
        X_list_to_transform=[X_all_orig, X_unseen_orig],
        K=cfg.K, alpha=cfg.alpha, beta=cfg.beta, seed=seed
    )
    met_u, phi_u_aligned, W_all_u_aligned = _align_and_doc_theta_metrics(
        phi_true=phi_true, phi_hat=phi_u, W_docs=W_all_u, Theta_true=Theta_all_true
    )
    perm_u, _, _ = align_topics(phi_true, phi_u)
    W_unseen_u_aligned = W_unseen_u[:, perm_u]

    # Augmented full
    lda_a, phi_a, [W_all_a, W_unseen_a] = fit_lda_and_transform_many(
        X_train=X_aug_train,
        X_list_to_transform=[X_all_orig, X_unseen_orig],
        K=cfg.K, alpha=cfg.alpha, beta=cfg.beta, seed=seed
    )
    met_a, phi_a_aligned, W_all_a_aligned = _align_and_doc_theta_metrics(
        phi_true=phi_true, phi_hat=phi_a, W_docs=W_all_a, Theta_true=Theta_all_true
    )
    perm_a, _, _ = align_topics(phi_true, phi_a)
    W_unseen_a_aligned = W_unseen_a[:, perm_a]

    # Topic drift (mean per-topic norms)
    met_u["l1_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_u_aligned, ord=1, axis=1)))
    met_u["l2_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_u_aligned, ord=2, axis=1)))
    met_a["l1_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_a_aligned, ord=1, axis=1)))
    met_a["l2_drift_topic"] = float(np.mean(np.linalg.norm(phi_true - phi_a_aligned, ord=2, axis=1)))

    lda_results = {
        "unaug_full": met_u,
        "aug_full":   met_a,
    }

    # ---------- Downstream tasks on completely unseen docs ----------
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

        for tag, W_all, W_unseen in [
            ("unaug_full", W_all_u_aligned, W_unseen_u_aligned),
            ("aug_full",   W_all_a_aligned, W_unseen_a_aligned),
        ]:
            # regression
            reg_model = Ridge(alpha=1.0, random_state=seed)
            reg_model.fit(W_all, y_reg_all)
            y_pred = reg_model.predict(W_unseen)
            mse = mean_squared_error(y_reg_unseen, y_pred)
            r2  = r2_score(y_reg_unseen, y_pred)

            # classification
            clf_model = LogisticRegression(max_iter=1000, random_state=seed)
            clf_model.fit(W_all, y_clf_all)
            y_proba = clf_model.predict_proba(W_unseen)[:, 1]
            y_pred_cls = (y_proba >= 0.5).astype(int)
            auc = roc_auc_score(y_clf_unseen, y_proba)
            acc = accuracy_score(y_clf_unseen, y_pred_cls)

            results_runs.append({
                "bucket": tag,
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

    # Add new full-corpus buckets
    for bucket in ["unaug_full", "aug_full"]:
        if bucket in lda_results:
            row[f"topic_jsd_{bucket}"] = float(lda_results[bucket]["phi_mean_jsd"])
            row[f"topic_cos_{bucket}"] = float(lda_results[bucket]["phi_mean_cos"])
            row[f"theta_l1_{bucket}"]  = float(lda_results[bucket]["theta_l1_mean"])
            row[f"theta_l2_{bucket}"]  = float(lda_results[bucket]["theta_l2_mean"])
            # topic drift norms if present
            if "l1_drift_topic" in lda_results[bucket]:
                row[f"phi_l1_{bucket}"] = float(lda_results[bucket]["l1_drift_topic"])
            if "l2_drift_topic" in lda_results[bucket]:
                row[f"phi_l2_{bucket}"] = float(lda_results[bucket]["l2_drift_topic"])

        if bucket in downstream_metrics:
            row[f"reg_mse_{bucket}"] = float(downstream_metrics[bucket]["reg_mse"])
            row[f"reg_r2_{bucket}"]  = float(downstream_metrics[bucket]["reg_r2"])
            row[f"clf_auc_{bucket}"] = float(downstream_metrics[bucket]["clf_auc"])
            row[f"clf_acc_{bucket}"] = float(downstream_metrics[bucket]["clf_acc"])

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
      5) Calibrate conditional thresholds on CALIB; select on EVAL
      6) Evaluate selection metrics; return artifacts
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

    # Compute correlations on calib (for monitoring)
    calib_units = [u for u in aug_units if u.doc_idx in set(idx_calib)]
    X_calib, y_calib, _ = assemble_feature_label_arrays_doclevel(
        calib_units, idf, phi, target="observed", oracle_mode="cosine", as_log=False
    )
    _, y_calib_oracle, _ = assemble_feature_label_arrays_doclevel(
        calib_units, idf, phi, target="oracle", oracle_mode="cosine", as_log=False
    )
    corr_calib_ahat   = float(np.corrcoef(reg.predict(X_calib), y_calib)[0, 1])
    corr_calib_oracle = float(np.corrcoef(reg.predict(X_calib), y_calib_oracle)[0, 1])
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
        idf=idf,
        phi=phi,
        lambda_obs=cfg.lambda_obs,
        alpha_cp=cfg.alpha_cp,
        rho=cfg.rho,
        verbose=True,
    )

    # Per-doc denominators for accept-rate on eval
    doc_total_eval: Dict[int, int] = {}
    for u in eval_units:
        doc_total_eval[u.doc_idx] = doc_total_eval.get(u.doc_idx, 0) + 1

    # Selection metrics; optionally include full-corpus LDA comparison (eval acceptances only here)
    cc_metrics = evaluate_selected_doclevel(
        selected_by_doc,
        doc_total_eval,
        phi,
        lambda_obs=cfg.lambda_obs,
        rho=cfg.rho,
        full_corpus_lda=True,                      # compare unaug vs (unaug + eval-accepts)
        docs=docs,
        splits={"idx_train": idx_train, "idx_calib": idx_calib, "idx_aug": idx_aug},
        seed=cfg.seed
    )

    # Baselines on eval
    selected_unfiltered_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        selected_unfiltered_by_doc.setdefault(u.doc_idx, []).append(u)

    baseline_unfiltered = evaluate_selected_doclevel(
        selected_unfiltered_by_doc, doc_total_eval, phi, cfg.lambda_obs, cfg.rho
    )

    selected_observed_by_doc: Dict[int, List[Unit]] = {}
    for u in eval_units:
        if float(u.A_obs_doc) >= float(cfg.lambda_obs):
            selected_observed_by_doc.setdefault(u.doc_idx, []).append(u)

    baseline_observed = evaluate_selected_doclevel(
        selected_observed_by_doc, doc_total_eval, phi, cfg.lambda_obs, cfg.rho
    )

    # Pretty print
    n_sel = sum(len(v) for v in selected_by_doc.values())
    n_tot = sum(doc_total_eval.values())
    if verbose:
        print(f"[CC] Selected generations: {n_sel} / {n_tot} ({100.0 * n_sel / max(1, n_tot):.1f}%)")
        print(f"[CC] Metrics (conditional): {cc_metrics}")
        print(f"[BASE] Unfiltered: {baseline_unfiltered}")
        print(f"[BASE] Observed  : {baseline_observed}")

    # Package results
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

                res = run_synthetic_experiment(cfg)
                print(f"Config: {cfg}, Results: {res['conditional']['metrics']}")

                # Full-corpus LDA + downstream on unseen set
                lda_results, downstream_metrics = evaluate_lda_and_downstream(cfg, res, seed=cfg.seed)

                row = results_to_row(cfg, res, lda_results, downstream_metrics)
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                out_path = os.path.join("results", f"synthetic_results_llm_cp_{name_exp}.csv")
                results_df.to_csv(out_path, index=False)

                print(results_df.tail(1))  # last row just added
                print("Done")
