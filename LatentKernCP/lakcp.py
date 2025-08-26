import numpy as np

from LatentKernCP.utils import *
from LatentKernCP.lambda_sqkr import lambda_path
from LatentKernCP.S_sqkr import S_path


class LAKCP:
    """
    Latent Kernel Conformal Pipeline:
      1) (Optional) search γ over a grid and, for each γ, compute a λ-path via `lambda_path`
      2) pick the best (γ, λ) by CSIC
      3) for each test point, run `S_path` to get S_opt

    Parameters
    ----------
    alpha : float
        Quantile level (e.g., 0.1).
    max_steps : int
        Max steps for path solvers.
    eps : float
        Small S offset for S_path initializer.
    tol : float
        Numerical tolerance.
    thres : float
        Small-step threshold for λ early stop in lambda_path.
    ridge : float
        Small ℓ2 regularizer for QP subproblems.
    start_side : {'left','right'}
        Side to start S_path from.
    gamma : float or None
        If provided, use this γ (skip grid search).
    gamma_grid : array-like
        Grid of γ values to search if `gamma` is None.
    verbose : bool
        Verbose logging.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_steps: int = 500,
        eps: float = 1e-3,
        tol: float = 1e-6,
        thres: float = 10.0,
        ridge: float = 1e-6,
        start_side: str = "left",
        gamma = None,
        gamma_grid: np.ndarray = np.logspace(0, 2, 50),
        verbose: bool = False,
    ):
        self.alpha = float(alpha)
        self.max_steps = int(max_steps)
        self.eps = float(eps)
        self.tol = float(tol)
        self.thres = float(thres)
        self.ridge = float(ridge)
        self.start_side = str(start_side)
        self.gamma = None if gamma is None else float(gamma)
        self.gamma_grid = np.asarray(gamma_grid, dtype=float)
        self.verbose = bool(verbose)

        self.lam = None
        self.calib_v = None
        self.calib_eta = None
        self.sics = None
        self.best_idx = None

    @staticmethod
    def _as_np_unique_sorted(a):
        a = np.asarray(a, dtype=int).ravel()
        return np.unique(a) if a.size else a

    # --------------------------------------------
    # Stage 1: search (or fix) gamma and best λ
    # --------------------------------------------
    def search_gamma_lambda(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray):
        """
        Runs lambda_path (for each γ if needed), selects best λ by CSIC,
        and stores: self.gamma, self.lam, self.calib_v, self.calib_eta, self.sics.
        """
        S_cal = np.asarray(S_cal, float).ravel()
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)

        if self.gamma is not None:
            K = kernel(X_cal, X_cal, self.gamma)
            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, self.alpha,
                max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                ridge=self.ridge, verbose=self.verbose
            )

            best = int(np.argmin(res["Csic"]))
            self.lam = float(res["lambdas"][best])
            self.calib_v = res["v_arr"][best, :].copy()
            self.calib_eta = res["eta_arr"][best, :].copy() if res.get("eta_arr") is not None else None
            self.sics = np.array([res["Csic"][best]], dtype=float)
            self.best_idx = best

        # grid search over gamma
        best_sic = np.inf
        best_gamma = None
        best_v = None
        best_eta = None
        best_lambda = None
        best_idx = None
        all_sics = []

        for g in self.gamma_grid:
            #print(f"[gamma search] trying γ={g:.6g}...")
            K = kernel(X_cal, X_cal, g)
            res = lambda_path(
                S_cal.ravel(), Phi_cal, K, self.alpha,
                max_steps=self.max_steps, tol=self.tol, thres=self.thres,
                ridge=self.ridge, verbose=False
            )
            #print(res)
            b = int(np.argmin(res["Csic"]))
            all_sics.append(float(res["Csic"][b]))
            #print(res["Csic"])
            #print(f"[gamma search] γ={g:.6g}, best λ*={res['lambdas'][b]:.6g}, CSIC={res['Csic'][b]:.6g}")

            if res["Csic"][b] < best_sic:
                best_sic = float(res["Csic"][b])
                best_gamma = float(g)
                best_v = res["v_arr"][b, :].copy()
                best_eta = res["eta_arr"][b, :].copy() if res.get("eta_arr") is not None else None
                best_lambda = float(res["lambdas"][b])
                best_idx = b

        # store
        self.gamma = best_gamma
        self.sics = np.array(all_sics, dtype=float)
        self.lam = best_lambda
        self.calib_v = best_v
        self.calib_eta = best_eta
        self.best_idx = best_idx

        if self.verbose:
            print(f"[gamma search] best γ={best_gamma:.6g}, λ*={best_lambda:.6g}, CSIC={best_sic:.6g}")


    def search_gamma_lambda_CV(self, X, Phi, S, gamma_grid, n_folds = 5,
                               random_state = 123):
        """
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        opt_lambdas = []   # (n_gamma,)
        opt_valid_err = [] # (n_gamma,)
        opt_test_err = []  # (n_gamma,)
        opt_test_fit = []  # (n_gamma, n_test)
        opt_train_fit = [] # (n_gamma, n_train)
        opt_miscoverage = [] # (n_gamma,)
        opt_lengths = []     # (n_gamma,)

        tau = 1 - self.alpha

        for gamma in gamma_grid:
            fold_val_err = []
            fold_lambdas = []
            fold_v = []
            fold_eta = []

            # --- CV loop ---
            for train_idx, valid_idx in kf.split(S):
                S_train, S_val = S[train_idx], S[valid_idx]
                Phi_train, Phi_val = Phi[train_idx, :], Phi[valid_idx, :]
                X_train, X_val = X[train_idx, :], X[valid_idx, :]

                # fit path
                K_train = kernel(X_train, X_train, gamma)
                res = lambda_path(S_train.ravel(), Phi_train, K_train, self.alpha,
                                max_steps=200, tol=1e-3, thres=10.0, verbose=False)

                # validation fit
                K_val = kernel(X_val, X_train, gamma)
                fit_val = Phi_val @ res['eta_arr'].T + (K_val @ res['v_arr'].T)/res['lambdas'][None, :]

                # quantile loss
                diff = S_val[:, None] - fit_val
                val_loss = np.where(diff > 0, tau * diff, (tau - 1) * diff)
                val_err = np.mean(val_loss, axis=0)

                # best lambda in this fold
                opt = np.argmin(val_err)
                fold_val_err.append(val_err[opt])
                fold_lambdas.append(res['lambdas'][opt])
                fold_v.append(res['v_arr'][opt, :])
                fold_eta.append(res['eta_arr'][opt, :])

            # --- aggregate across folds ---
            mean_val_err = np.mean(fold_val_err)
            best_fold = np.argmin(fold_val_err)

            opt_valid_err.append(mean_val_err)
            opt_lambdas.append(fold_lambdas[best_fold])
            opt_v = fold_v[best_fold]
            opt_eta = fold_eta[best_fold]

            # retrain on all data with chosen lambda/gamma if needed
            K_full = kernel(X, X, gamma)
            res_full = lambda_path(S.ravel(), Phi, K_full, self.alpha,
                                max_steps=200, tol=1e-3, thres=10.0, verbose=False)
            # choose lambda closest to fold choice
            idx = np.argmin(np.abs(res_full['lambdas'] - opt_lambdas[-1]))
            opt_train_fit.append(res_full['fit'][idx, :])

        # choose best gamma by CV error
        opt_idx = np.argmin(opt_valid_err)
        best_gamma = gamma_grid[opt_idx]
        best_lambda = opt_lambdas[opt_idx]
        return {
            'best_gamma': best_gamma,
            'best_lambda': best_lambda,
            'opt_idx': opt_idx,
            'opt_valid_err': opt_valid_err,
            'opt_train_fit': opt_train_fit,
            'opt_lambdas': opt_lambdas,
        }

    # -------------------------------------------------
    # Stage 2: compute S_opt for each test point (S_path)
    # -------------------------------------------------
    def fit(self, X_cal: np.ndarray, Phi_cal: np.ndarray, S_cal: np.ndarray,
            X_test: np.ndarray, Phi_test: np.ndarray) -> np.ndarray:
        """
        Ensure gamma/lambda are selected, then run S_path for each test point.

        Returns
        -------
        S_opt_array : (n_test,) array of optimal S for each test sample.
        """
        # ensure we have gamma, lambda, v, eta
        if self.lam is None or self.gamma is None or self.calib_v is None:
            self.search_gamma_lambda(X_cal, Phi_cal, S_cal)

        S_cal = np.asarray(S_cal, float).ravel()
        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        X_test = np.asarray(X_test, float)
        Phi_test = np.asarray(Phi_test, float)

        n_test = X_test.shape[0]
        out_S = np.empty(n_test, dtype=float)

        for i in range(n_test):
            x_row = X_test[i].reshape(1, -1)          # (1, p)
            phi_row = Phi_test[i].reshape(1, -1)      # (1, d)

            X_all = np.vstack([X_cal, x_row])         # ((n_cal+1), p)
            Phi_all = np.vstack([Phi_cal, phi_row])   # ((n_cal+1), d)
            K_all = kernel(X_all, X_all, self.gamma)

            res_S = S_path(
                S_cal, Phi_all, K_all, self.lam, self.alpha, self.calib_v,
                start_side=self.start_side, max_steps=self.max_steps,
                eps=self.eps, tol=self.tol, ridge=self.ridge, verbose=self.verbose
            )
            out_S[i] = float(res_S["S_opt"])

        return out_S

    # -------------------------
    # Convenience: predict fit
    # -------------------------
    def predict(self, X_cal, Phi_cal, X_new, Phi_new) -> np.ndarray:
        """
        Given calibrated (γ, λ, v, η), compute g_hat for new X_new, Phi_new.
        Note: requires that `search_gamma_lambda` has been run.

        Returns
        -------
        g_new : (n_new,) model fit at λ* using stored (v, η).
        """
        if self.lam is None or self.gamma is None or self.calib_v is None:
            raise RuntimeError("Call `search_gamma_lambda` or `fit` first to set (γ, λ, v, η).")

        X_cal = np.asarray(X_cal, float)
        Phi_cal = np.asarray(Phi_cal, float)
        X_new = np.asarray(X_new, float)
        Phi_new = np.asarray(Phi_new, float)

        K_new = kernel(X_new, X_cal, self.gamma)              # (n_new, n_cal)
        Kv = K_new @ self.calib_v                             # (n_new,)
        if self.calib_eta is None:
            eta_term = 0.0
        else:
            eta_term = Phi_new @ self.calib_eta               # (n_new,)

        return eta_term + Kv / self.lam