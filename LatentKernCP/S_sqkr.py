import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel
from LatentKernCP.utils import kernel, pinball

def _as_np_unique_sorted(a):
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0: return a
    return np.unique(a)

def S_path(S_cal, Phi, K, lam, alpha, best_v, 
           start_side='left', max_steps=200, eps=1e-1, tol=1e-6, ridge=1e-8, verbose=False):
    S_cal = np.asarray(S_cal, dtype=float).ravel()
    n = S_cal.size + 1
    K = np.asarray(K, dtype=float)
    
    if Phi is not None:
        Phi = np.asarray(Phi, dtype=float)
        d = Phi.shape[1]
    else:
        d = 0
    
    # --- init ---
    ini   = S_init(S_cal, Phi, K, lam, alpha, best_v, start_side, eps, tol)
    indE  = _as_np_unique_sorted(ini["indE"])
    indL  = _as_np_unique_sorted(ini["indL"])
    indR  = _as_np_unique_sorted(ini["indR"])
    S     = float(ini["S"]) #np.min(S_cal) #float(ini["S"]) + tol
    v     = np.asarray(ini["v"],   dtype=float).copy()
    eta   = np.asarray(ini["eta"], dtype=float).copy() if d > 0 else float(ini["eta"])
    
    S_ini = S
    if verbose:
        #print(f"v: {v}, eta: {eta}")
        print(f"[init] S={S:.6g}; |E|={indE.size}, |L|={indL.size}, |R|={indR.size}")

    # storage
    eta_arr = np.zeros((max_steps+1, d))
    v_arr   = np.zeros((max_steps+1, n))
    Csic    = np.zeros(max_steps+1)
    fit     = np.zeros((max_steps+1, n))
    S_vals  = np.zeros(max_steps+1)
    Elbows  = [None]*(max_steps+1)

    eta_arr[0] = eta
    v_arr[0] = v
    S_vals[0] = S
    Elbows[0] = indE.copy()

    # current fit
    Kv = K @ v
    if d > 0:
        g_hat = Phi @ eta + Kv / lam
    else:
        g_hat = eta + Kv / lam
    
    S_vec = np.append(S_cal, S)
    r = S_vec - g_hat
    
    csic = np.log(pinball(g_hat, S_vec, 1-alpha) / n) + (np.log(n) / (2 * n)) * indE.size
    Csic[0] = csic
    fit[0] = g_hat

    k = 0
    cache = {}
    while k < max_steps:
        k += 1
        if (indL.size == 0 and indR.size == 0) or (n-1 in indR):
            if verbose:
                why = "L and R empty" if (indL.size==0 and indR.size==0) else "test moved to R"
                print(f"[stop] {why}")
            break
        
        # ----- Step 1: Solve elbow linear system -----
        E = indE
        m = E.size

        PhiE = Phi[E, :]
        PhiE_mult = np.linalg.inv(PhiE.T @ PhiE) 
        KEE  = K[np.ix_(E, E)]

        key = tuple(E.tolist())
        if key in cache:
            PiE, A_dag, FEpinv = cache[key]
        else:
            FEpinv = np.linalg.inv(PhiE.T @ PhiE) 
            PiE = np.eye(m) - PhiE @ PhiE_mult @ PhiE.T
            A   = PiE @ KEE @ PiE
            A_dag = np.linalg.pinv(A, rcond=1e-12)
            cache[key] = (PiE, A_dag, FEpinv)

        e_test = np.zeros(m); e_test[indE == n-1] = 1.0

        dvE_dS = lam * A_dag @ (PiE @ e_test)          # (|E|,)
        rhs_deta = PhiE.T @ (e_test - KEE @ dvE_dS / lam)
        deta_dS  = np.linalg.solve(PhiE.T @ PhiE, rhs_deta)
        dg_dS    = Phi @ deta_dS + (K[:, indE] @ dvE_dS) / lam
        dr_dS    = -dg_dS
        dr_dS[n-1] += 1.0 

        # ----- Step 2: Find the next S and event -----
        cand_steps = []
        cand_who = []
        cand_dir = []

        # (a) point in E hits a bound: v_E -> -alpha or 1-alpha
        # (a) any elbow coordinate hits either bound: v_E -> {-alpha, 1-alpha}
        vE = v[E].astype(float)
        for loc in range(m):
            slope = dvE_dS[loc]
            if abs(slope) <= tol:
                continue
            for bnd, side in [(-alpha, 'L'), (1.0 - alpha, 'R')]:
                t = (bnd - vE[loc]) / slope
                if t > tol:
                    cand_steps.append(float(t))
                    cand_who.append(('leave', int(E[loc])))
                    cand_dir.append(side)

        # (b) residual in L∪R becomes zero
        LR = np.concatenate((indL, indR))
        for i in LR:
            slope = dr_dS[i]
            if abs(slope) < tol:
                continue
            t = -r[i] / slope 
            if t > tol:
                cand_steps.append(float(t))
                cand_who.append(('hit', int(i)))
                cand_dir.append(None) 
                #cand_dir.append(int(np.sign(slope)))

        if not cand_steps:
            if verbose: print("[stop] no more candidate events")
            break

        cand_steps = np.array(cand_steps, float)              
        imin = int(np.argmin(cand_steps))
        step = cand_steps[imin]
        ev_kind, ev_idx = cand_who[imin]
        ev_dir = cand_dir[imin]

        if verbose:
            print(f"[{k}] event={ev_kind}, idx={ev_idx}, step={step:.6g}")

        # take step 
        S_next = S + step
        if verbose:
            print(f"[{k}] step to S={S_next:.6g}")

        # early stops
        if np.abs(S_next - S) < tol:
            if verbose:
                print(f"[stop] step too small, S={S_next:.6g}")
                #S = S_next
                k += 1
            break

        if S_next > np.max(S_cal) + eps:
            if verbose:
                print(f"[stop] exceeded max S_cal + eps, S={S_next:.6g}")
                #S = S_next
                k += 1
            break

        # ----- Step 3: Update the next E, L, R -----
        if ev_kind == 'leave':
            E_new = E[E != ev_idx]
            if ev_dir == 'L':
                 indL = _as_np_unique_sorted(np.append(indL, ev_idx))
            else:  # 'R'
                indR = _as_np_unique_sorted(np.append(indR, ev_idx))
            indE = _as_np_unique_sorted(E_new)
        else:                         
            if ev_idx in indL:
                indL = indL[indL != ev_idx]
            else:
                indR = indR[indR != ev_idx]
            indE = _as_np_unique_sorted(np.append(indE, ev_idx))

        if ev_idx == n-1:
            if verbose: 
                print(f"[stop] exited elbow at S={S_next:.6g}")
                S = S_next
            break

        # ----- Step 4: Update v and eta -----
        v = np.zeros(n, dtype=float)
        v[indL] = -alpha
        v[indR] = 1 - alpha

        E = indE.copy()
        m = len(indE)

        KEE = K[np.ix_(E, E)]
        PhiE = Phi[E, :]
        S_vec = np.append(S_cal, S_next)
        S_E  = np.asarray(S_vec[E],  float).ravel()

        # Build the stacked linear operator H z ≈ b  (least-squares)
        # H = [[PhiE, KEE], [0, PhiE^T]], b = [S_E, 0]
        # top = np.hstack([PhiE, KEE / lam])
        # bot = np.hstack([np.zeros((d, d)), PhiE.T])
        # H   = np.vstack([top, bot])
        # b   = np.concatenate([S_E, np.zeros(d)]
        #         # Build the stacked linear operator H z ≈ b  (least-squares)
        # H = [[PhiE, KEE/lam], [0, PhiE^T]]
        # RHS includes L/R offsets (top) and KKT feature RHS (bottom)
        one_L = np.ones(len(indL))
        one_R = np.ones(len(indR))
        dE = ((-alpha) * (K[np.ix_(E, indL)] @ one_L) +
              (1.0 - alpha) * (K[np.ix_(E, indR)] @ one_R)) / lam
        top = np.hstack([PhiE, KEE / lam])
        bot = np.hstack([np.zeros((d, d)), PhiE.T])
        H   = np.vstack([top, bot])
        b_top = S_E - dE
        b_bot = (alpha * (Phi[indL, :].T @ one_L) -
                 (1.0 - alpha) * (Phi[indR, :].T @ one_R)) if d > 0 else np.zeros(d)
        b     = np.concatenate([b_top, b_bot])

        # QP: 0.5 z^T P z + q^T z, with P = H^T H + ridge I, q = -H^T b
        P_np = H.T @ H + ridge * np.eye(d + m)
        q_np = -(H.T @ b)

        # Inequalities: box on v_E only
        #  v_E ≤ (1-α) → [0_dxm  I_m] z ≤ (1-α)·1
        # -v_E ≤ α     → [0_dxm -I_m] z ≤ α·1
        Zdm = np.zeros((m, d))
        G_np = np.vstack([
            np.hstack([Zdm,  np.eye(m)]),
            np.hstack([Zdm, -np.eye(m)])
        ])
        h_np = np.hstack([
            (1.0 - alpha) * np.ones(m),
            alpha * np.ones(m)
        ])

        # Convert to cvxopt types
        P = matrix(P_np, tc='d')
        q = matrix(q_np, tc='d')
        G = matrix(G_np, tc='d')
        h = matrix(h_np, tc='d')

        Aeq, beq = None, None
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, Aeq, beq)

        if sol['status'] == 'optimal':
            delta = np.array(sol['x']).flatten()
        else:
            print("Warning: CVXOPT did not find an optimal solution. Status:", sol['status'])
    
        z = np.array(sol['x']).ravel()
        eta = z[:d]
        vE  = z[d:]
        v[E] = vE

        # update fit
        Kv = K @ v
        if d > 0:
            g_hat = Phi @ eta + Kv / lam
        else:
            g_hat = eta + Kv / lam
        
        # store
        fit[k]     = g_hat
        S_vals[k]  = S_next
        v_arr[k]   = v
        eta_arr[k] = eta
        Elbows[k]  = indE.copy()
        Csic[k]    = np.log(pinball(g_hat, np.append(S_cal, S_next), 1-alpha) / n) + (np.log(n)/(2*n))*indE.size

        S = S_next

    #S_opt = S_vals[np.argmin(Csic[:k])]

    if verbose:
        print(f"[done] steps={k}, |E|={indE.size}, S={S:.6g}")

    return {
        "S_vals": S_vals[:k],
        "v_arr":  v_arr[:k],
        "eta_arr":eta_arr[:k],
        "Elbows": Elbows[:k],
        "fit":   fit[:k],
        "Csic":  Csic[:k],
        "steps": k,
        "S_opt": S,
        "S_init": S_ini
    }


def S_init(S_cal, Phi, K, lam, alpha, opt_v_lambda,
               start_side='left', eps=1e-1, tol=1e-6, ridge=1e-8, verbose=False):
    
    # Initialize sets
    indE = np.where((opt_v_lambda > -alpha + tol) & (opt_v_lambda < 1.0 - alpha - tol))[0].tolist()
    indL = np.where(np.abs(opt_v_lambda + alpha) <= tol)[0].tolist()
    indR = np.where(np.abs(opt_v_lambda - (1.0 - alpha)) <= tol)[0].tolist()

    n = len(S_cal)
    d = Phi.shape[1]
    E = indE.copy()
    m = len(E)

    v = np.zeros(n, dtype=float)
    v[indL] = -alpha
    v[indR] = 1 - alpha

    alpha0 = -alpha if start_side == 'left' else 1-alpha

    KEE, kEn = K[np.ix_(E, E)], K[indE, n]
    KEL, KER = K[np.ix_(E, indL)], K[np.ix_(E, indR)]
    PhiE, Phin = Phi[indE, :], Phi[n]
    PhiL, PhiR = Phi[indL, :], Phi[indR, :]
    one_L = np.ones(len(indL))
    one_R = np.ones(len(indR))

    # Top (correct signs)
    S_E = S_cal[indE]
    S_E_eff = S_E - (alpha0 * kEn - alpha * (KEL @ one_L) + (1 - alpha) * (KER @ one_R)) / lam
    top = np.hstack([PhiE, KEE / lam])
    b_top = S_E_eff

    # Bottom (transpose + sign)
    bot = np.hstack([np.zeros((d, d)), PhiE.T])
    b_bot = -alpha0 * Phin + alpha * (PhiL.T @ one_L) - (1 - alpha) * (PhiR.T @ one_R)

    H = np.vstack([top, bot])
    b = np.concatenate([b_top, b_bot])

    # QP matrices sized with m, not n
    P_np = H.T @ H + ridge * np.eye(d + m)
    q_np = -(H.T @ b)
    Zdm = np.zeros((m, d))
    G_np = np.vstack([
        np.hstack([Zdm,  np.eye(m)]),
        np.hstack([Zdm, -np.eye(m)])
    ])
    h_np = np.hstack([
        (1.0 - alpha) * np.ones(m),
        alpha * np.ones(m)
    ])

    # Convert to cvxopt types
    P = matrix(P_np, tc='d')
    q = matrix(q_np, tc='d')
    G = matrix(G_np, tc='d')
    h = matrix(h_np, tc='d')

    Aeq, beq = None, None
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, Aeq, beq)

    if sol['status'] == 'optimal':
        delta = np.array(sol['x']).flatten()
    else:
        print("Warning: CVXOPT did not find an optimal solution. Status:", sol['status'])

    z = np.array(sol['x']).ravel()
    eta_init = z[:d]
    delta  = z[d:]
    v[E] = delta

    v_init = np.append(v, alpha0)
    g_hat = Phi @ eta_init + (K @ v_init) / lam
    S_init = g_hat[-1]

    indE.append(n) # start with n+1 in indE
    
    return {
        "v": v_init,
        "eta": eta_init,
        "S": S_init,
        "indE": indE,
        "indR": indR,
        "indL": indL
    }