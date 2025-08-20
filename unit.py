# ------------------------------
# Doc-level Units
# ------------------------------
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from config import SynthConfig, Document, Candidate, dirichlet_sample, softmax_temp
from helper_similarity_metrics import observed_A, oracle_A_star
from featurize_document import featurize_candidate_doclevel, compute_idf

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

        # Do not pass phi & u.doc_theta so topic-alignment features are inactive
        x = featurize_candidate_doclevel(u.masked_true, u.candidates, u.doc_ctx, idf,
                                         phi=None, doc_theta=None)
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
