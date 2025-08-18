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


from config import SynthConfig, Document, Candidate, make_topics, generate_corpus, softmax_temp, dirichlet_sample, assemble_feature_label_arrays_doclevel
from unit import BaseUnit, Unit, assemble_augmented_docs_and_units, gen_candidates_for_mask
from helper_similarity_metrics import cosine_from_counts, cosine_from_freq, counts_vec, distinct_1, js_divergence
from cp_selection import per_doc_S_doclevel, per_doc_S_doclevel_multi, global_threshold_S_doclevel, fit_conditional_threshold_doclevel
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
                "name": name,
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



