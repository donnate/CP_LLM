from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Sequence
import numpy as np



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

