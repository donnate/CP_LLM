
# Joint LDS on a graph via TV (genlasso)


library(genlasso)
library(igraph)
library(Matrix)



incidence_from_graph <- function(g) {
  Elist <- as_edgelist(g, names = FALSE)
  m <- vcount(g); E <- nrow(Elist)
  D <- matrix(0, nrow = E, ncol = m)
  for (e in 1:E) {
    i <- Elist[e, 1]; j <- Elist[e, 2]
    D[e, i] <- -1; D[e, j] <- 1
  }
  D
}


simulate_lti <- function(A, T, x0 = NULL) {
  ### x_{t+1} = A x_t + N(0, I_d)
  d <- nrow(A)
  X <- matrix(0, nrow = d, ncol = T + 1)
  if (!is.null(x0)) X[, 1] <- x0
  for (t in 1:T) {
    X[, t + 1] <- A %*% X[, t] + rnorm(d)
  }
  X
}


ts_split <- function(X, T_train, T_val) {
  ### train test val split
  Ttot <- ncol(X) - 1
  stopifnot(T_train + T_val < Ttot)
  T_test <- Ttot - T_train - T_val
  Xtr <- X[, 1:T_train, drop = FALSE];       Ytr <- X[, 2:(T_train + 1), drop = FALSE]
  Xva <- X[, (T_train + 1):(T_train + T_val), drop = FALSE]
  Yva <- X[, (T_train + 2):(T_train + T_val + 1), drop = FALSE]
  Xte <- X[, (T_train + T_val + 1):Ttot, drop = FALSE]
  Yte <- X[, (T_train + T_val + 2):(Ttot + 1), drop = FALSE]
  list(Xtr = Xtr, Ytr = Ytr, Xva = Xva, Yva = Yva, Xte = Xte, Yte = Yte)
}


build_Qy <- function(X_list, Y_list, scale_m = TRUE) {
  ### define the matrix Q and stacked responses (as in the paper)
  m <- length(X_list)
  d <- nrow(X_list[[1]])
  Q_list <- vector("list", m)
  y_vec <- numeric(0)
  for (l in 1:m) {
    Q_list[[l]] <- kronecker(t(X_list[[l]]), diag(d))    # (T*d) x d^2
    y_vec <- c(y_vec, as.vector(Y_list[[l]]))            # vec(Y_l)
  }
  Q <- as.matrix(bdiag(Q_list))                           # (m*T*d) x (m*d^2)
  if (scale_m) { Q <- Q / sqrt(m); y_vec <- y_vec / sqrt(m) }
  list(Q = Q, y = y_vec)
}


beta_to_Alist <- function(beta, m, d) {
  # from beta ( m*d^2) to list of m A_l (d x d)
  d2 <- d * d
  Alist <- vector("list", m)
  for (l in 1:m) {
    idx <- ((l - 1) * d2 + 1):(l * d2)
    Alist[[l]] <- matrix(beta[idx], nrow = d, ncol = d)
  }
  Alist
}

# Baselines: independent OLS per node; pooled OLS across nodes
fit_ols_per_node <- function(X_list, Y_list, ridge = 1e-6) {
  m <- length(X_list); d <- nrow(X_list[[1]])
  Alist <- vector("list", m)
  for (l in 1:m) {
    X <- X_list[[l]]; Y <- Y_list[[l]]
    XXt <- X %*% t(X)
    Alist[[l]] <- Y %*% t(X) %*% solve(XXt + ridge * diag(d))
  }
  Alist
}
fit_pooled_ols <- function(X_list, Y_list, ridge = 1e-6) {
  d <- nrow(X_list[[1]])
  Xall <- do.call(cbind, X_list); Yall <- do.call(cbind, Y_list)
  XXt <- Xall %*% t(Xall)
  A <- Yall %*% t(Xall) %*% solve(XXt + ridge * diag(d))
  replicate(length(X_list), A, simplify = FALSE)
}

# Metrics
A_mse <- function(Ahat_list, Atrue_list) {
  mean(sapply(1:length(Atrue_list), function(l) sum((Ahat_list[[l]] - Atrue_list[[l]])^2)))
}
pred_mse <- function(Ahat_list, X_list, Y_list) {
  num <- 0; den <- 0
  for (l in 1:length(Ahat_list)) {
    pred <- Ahat_list[[l]] %*% X_list[[l]]
    r <- as.vector(pred - Y_list[[l]])
    num <- num + sum(r^2); den <- den + length(r)
  }
  num / den
}

# Edge-wise Frobenius gaps (for simple boundary detection on path graphs)
edge_boundary_scores <- function(Ahat_list, edges, true_break_edges = NULL, top_k = NULL) {
  E <- nrow(edges)
  diffs <- numeric(E)
  for (e in 1:E) {
    i <- edges[e, 1]; j <- edges[e, 2]
    diffs[e] <- sqrt(sum((Ahat_list[[i]] - Ahat_list[[j]])^2))
  }
  ord <- order(diffs, decreasing = TRUE)
  out <- list(edge_diff = diffs, ranking = ord)
  if (!is.null(true_break_edges) && !is.null(top_k)) {
    sel <- sort(ord[1:top_k])
    prec <- length(intersect(sel, true_break_edges)) / top_k
    rec  <- length(intersect(sel, true_break_edges)) / length(true_break_edges)
    out$precision_at_k <- prec; out$recall_at_k <- rec; out$selected <- sel
  }
  out
}

# ---------- TV-LDS fitter via genlasso with hold-out selection ----------
fit_tv_lds <- function(
    Xtr_list, Ytr_list,
    Xva_list = NULL, Yva_list = NULL,
    D_graph, d, maxsteps = 200
) {
  m <- length(Xtr_list); d2 <- d * d
  De <- kronecker(D_graph, diag(d2))
  des_tr <- build_Qy(Xtr_list, Ytr_list, scale_m = TRUE)
  
  fit <- genlasso(y = des_tr$y, X = des_tr$Q, D = De, maxsteps = maxsteps)
  # Coefficients along the path (p x L)
  beta_mat <- if (!is.null(fit$beta)) fit$beta else as.matrix(coef(fit)$beta)
  lambdas  <- fit$lambda
  
  # If no validation provided, return the full path
  if (is.null(Xva_list)) {
    return(list(beta_path = beta_mat, lambda = lambdas, fit = fit))
  }
  
  # Validation selection
  des_va <- build_Qy(Xva_list, Yva_list, scale_m = TRUE)
  preds  <- des_va$Q %*% beta_mat                # (n_val) x L
  val_mse <- colMeans((preds - des_va$y)^2)      # average over entries
  
  j <- which.min(val_mse)
  beta_sel <- beta_mat[, j]
  Alist_sel <- beta_to_Alist(beta_sel, m, d)
  
  list(
    A = Alist_sel,
    beta = beta_sel,
    lambda = lambdas[j],
    val_mse = val_mse,
    lambda_path = lambdas
  )
}

# ---------- Experiment 1: Path graph, moderate T ----------
experiment_path_moderateT <- function(seed = 1) {
  set.seed(seed)
  m <- 50; d <- 2
  # Three piecewise-constant regions on a path
  breaks <- c(20, 35)  # boundaries at edges (20,21) and (35,36)
  groups <- rep(1, m)
  groups[(breaks[1] + 1):breaks[2]] <- 2
  groups[(breaks[2] + 1):m] <- 3
  
  A1 <- matrix(c(0.8, 0.1,
                 0.0, 0.6), nrow = d, byrow = TRUE)
  A2 <- matrix(c(0.6, -0.2,
                 0.1,  0.7), nrow = d, byrow = TRUE)
  A3 <- matrix(c(0.5,  0.0,
                 0.3,  0.4), nrow = d, byrow = TRUE)
  Atrue <- lapply(1:m, function(l) switch(groups[l], A1, A2, A3))
  
  # Graph: path
  edges <- cbind(1:(m - 1), 2:m)
  g <- graph_from_edgelist(edges, directed = FALSE)
  Dg <- incidence_from_graph(g)
  
  # Simulate trajectories
  Ttot <- 60; Ttr <- 40; Tva <- 10      # 10 left for test
  X_all <- vector("list", m)
  splits <- vector("list", m)
  for (l in 1:m) {
    X <- simulate_lti(Atrue[[l]], Ttot)
    X_all[[l]] <- X
    splits[[l]] <- ts_split(X, T_train = Ttr, T_val = Tva)
  }
  
  # Build (X,Y) lists per split
  Xtr_list <- lapply(splits, `[[`, "Xtr"); Ytr_list <- lapply(splits, `[[`, "Ytr")
  Xva_list <- lapply(splits, `[[`, "Xva"); Yva_list <- lapply(splits, `[[`, "Yva")
  Xte_list <- lapply(splits, `[[`, "Xte"); Yte_list <- lapply(splits, `[[`, "Yte")
  
  # TV estimator with val selection
  tv_fit <- fit_tv_lds(Xtr_list, Ytr_list, Xva_list, Yva_list, D_graph = Dg, d = d, maxsteps = 200)
  
  # Baselines
  ols_ind <- fit_ols_per_node(Xtr_list, Ytr_list)
  ols_pool <- fit_pooled_ols(Xtr_list, Ytr_list)
  
  # Metrics
  out <- list()
  out$lambda_selected <- tv_fit$lambda
  out$A_mse <- c(
    tv = A_mse(tv_fit$A, Atrue),
    ols_ind = A_mse(ols_ind, Atrue),
    ols_pool = A_mse(ols_pool, Atrue)
  )
  out$pred_mse_test <- c(
    tv = pred_mse(tv_fit$A, Xte_list, Yte_list),
    ols_ind = pred_mse(ols_ind, Xte_list, Yte_list),
    ols_pool = pred_mse(ols_pool, Xte_list, Yte_list)
  )
  
  # Simple boundary detection: top-2 edges by ||A_i - A_j||_F vs truth {20,35}
  eb <- edge_boundary_scores(tv_fit$A, edges, true_break_edges = c(20, 35), top_k = 2)
  out$boundary_precision_at_2 <- eb$precision_at_k
  out$boundary_recall_at_2 <- eb$recall_at_k
  out$selected_edges <- eb$selected
  out$val_curve <- data.frame(lambda = tv_fit$lambda_path, val_mse = tv_fit$val_mse)
  out
}

# ---------- Experiment 2: Path graph, short T (hard regime) ----------
experiment_path_shortT <- function(seed = 2) {
  set.seed(seed)
  m <- 50; d <- 2
  breaks <- c(20, 35)
  groups <- rep(1, m)
  groups[(breaks[1] + 1):breaks[2]] <- 2
  groups[(breaks[2] + 1):m] <- 3
  
  A1 <- matrix(c(0.8, 0.1, 0.0, 0.6), nrow = d, byrow = TRUE)
  A2 <- matrix(c(0.6, -0.2, 0.1, 0.7), nrow = d, byrow = TRUE)
  A3 <- matrix(c(0.5,  0.0, 0.3, 0.4), nrow = d, byrow = TRUE)
  Atrue <- lapply(1:m, function(l) switch(groups[l], A1, A2, A3))
  
  edges <- cbind(1:(m - 1), 2:m)
  g <- graph_from_edgelist(edges, directed = FALSE)
  Dg <- incidence_from_graph(g)
  
  # Much shorter series
  Ttot <- 16; Ttr <- 8; Tva <- 4         # 4 left for test
  X_all <- vector("list", m)
  splits <- vector("list", m)
  for (l in 1:m) {
    X <- simulate_lti(Atrue[[l]], Ttot)
    splits[[l]] <- ts_split(X, T_train = Ttr, T_val = Tva)
  }
  Xtr_list <- lapply(splits, `[[`, "Xtr"); Ytr_list <- lapply(splits, `[[`, "Ytr")
  Xva_list <- lapply(splits, `[[`, "Xva"); Yva_list <- lapply(splits, `[[`, "Yva")
  Xte_list <- lapply(splits, `[[`, "Xte"); Yte_list <- lapply(splits, `[[`, "Yte")
  
  tv_fit <- fit_tv_lds(Xtr_list, Ytr_list, Xva_list, Yva_list, D_graph = Dg, d = d, maxsteps = 200)
  ols_ind <- fit_ols_per_node(Xtr_list, Ytr_list)
  ols_pool <- fit_pooled_ols(Xtr_list, Ytr_list)
  
  out <- list()
  out$lambda_selected <- tv_fit$lambda
  out$A_mse <- c(
    tv = A_mse(tv_fit$A, Atrue),
    ols_ind = A_mse(ols_ind, Atrue),
    ols_pool = A_mse(ols_pool, Atrue)
  )
  out$pred_mse_test <- c(
    tv = pred_mse(tv_fit$A, Xte_list, Yte_list),
    ols_ind = pred_mse(ols_ind, Xte_list, Yte_list),
    ols_pool = pred_mse(ols_pool, Xte_list, Yte_list)
  )
  out$val_curve <- data.frame(lambda = tv_fit$lambda_path, val_mse = tv_fit$val_mse)
  out
}

# ---------- Run both experiments and print summaries ----------
run_experiments <- function() {
  cat("=== Experiment 1: Path, moderate T ===\n")
  e1 <- experiment_path_moderateT(seed = 1)
  print(list(lambda_selected = e1$lambda_selected,
             A_mse = e1$A_mse,
             pred_mse_test = e1$pred_mse_test,
             boundary_precision_at_2 = e1$boundary_precision_at_2,
             boundary_recall_at_2 = e1$boundary_recall_at_2,
             selected_edges = e1$selected_edges))
  cat("\n=== Experiment 2: Path, short T ===\n")
  e2 <- experiment_path_shortT(seed = 2)
  print(list(lambda_selected = e2$lambda_selected,
             A_mse = e2$A_mse,
             pred_mse_test = e2$pred_mse_test))
  invisible(list(exp1 = e1, exp2 = e2))
}

# Uncomment to run immediately:
results <- run_experiments()
