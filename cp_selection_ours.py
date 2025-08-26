import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GroupKFold
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import KFold


from LatentKernCP_.lakcp import LAKCP
from LatentKernCP_.utils import *
from unit import Unit, assemble_feature_label_arrays_doclevel  # Assuming Unit is defined in a module named 'unit'

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
    bad_scores = [u.A_hat for u in units_for_doc ]
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
    use_bad_only: bool = False,       # train CC on BAD-doc scores only
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
        yhat_d = yhat_c[idxs]   #### y observed
        bad_d  = (yobs_c[idxs] < float(lambda_obs))
        n_bad  = int(np.sum(bad_d))
        if n_bad <= rho:
            S_d = -np.inf
        else:
            #svals = np.sort(yhat_d[bad_d])        # ascending
            svals = np.sort(yhat_d)  
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


    Phi_cal = phi_fn_intercept(X_cc_fit)

    print("Starting LAKCP...")
    lakcp = LAKCP(alpha = alpha_cp,
                                max_steps = 1000,
                                eps = 1e-03,
                                tol = 1e-06,
                                thres = 10.0,
                                ridge = 1e-08,
                                start_side = 'left',
                                gamma = None,
                                gamma_grid = np.logspace(-4,2,15),
                                verbose=True)
    
    lakcp.search_gamma_lambda(X_cc_fit, Phi_cal, yhat_c.ravel())
    print("LAKCP finished choosing lambda and gamma.")

    # -- Predict thresholds per **test doc** and select units
    thresholds_by_doc: Dict[int, float] = {}
    selected_by_doc: Dict[int, List[Any]] = {}
    selected_units: List[Any] = []

    for d in test_doc_ids:
        x_doc = theta_by_doc[d][None, :]
        Phi_test = phi_fn_intercept(X_cc_fit)
        pred_set = lakcp.fit(X_cc_fit, Phi_cal, yhat_c.ravel(),
                            x_doc, Phi_test)
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
