# utils/plot_pack.py
import matplotlib.pyplot as plt
import numpy as np


def plot_recon_error_t(err_t: np.ndarray, title: str):
    plt.figure(figsize=(10, 3))
    plt.plot(err_t)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("MSE_t")


def plot_diff_heatmap(x: np.ndarray, x_cf: np.ndarray, title: str):
    diff = np.abs(x - x_cf)
    plt.figure(figsize=(9, 4))
    plt.imshow(diff.T, aspect="auto")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("feature")
    plt.colorbar()


# =========================
# OPTIMIZATION PLOTS (classic)
# =========================


def plot_objective_trace(best_per_gen: np.ndarray, title: str, ylabel: str):
    """
    best_per_gen: shape (G,) or (G, k) for k objectives.
    If (G,k), we'll plot each objective trace.
    """
    plt.figure(figsize=(8, 4))
    if best_per_gen.ndim == 1:
        plt.plot(best_per_gen)
    else:
        for j in range(best_per_gen.shape[1]):
            plt.plot(best_per_gen[:, j], label=f"obj{j}")
        plt.legend()
    plt.title(title)
    plt.xlabel("generation")
    plt.ylabel(ylabel)


def plot_constraint_violation_trace(cv_per_gen: np.ndarray, title: str):
    plt.figure(figsize=(8, 4))
    plt.plot(cv_per_gen)
    plt.title(title)
    plt.xlabel("generation")
    plt.ylabel("total constraint violation")


def plot_pareto_2d(F: np.ndarray, i: int, j: int, title: str, xlabel: str, ylabel: str):
    """
    F: (N, M) objective values
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(F[:, i], F[:, j], alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def plot_design_space_projection(X: np.ndarray, title: str, max_points: int = 2000):
    """
    'Design space' / decision space projection via PCA(2) without sklearn.
    X: (N, D)
    """
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], size=max_points, replace=False)
        Xp = X[idx]
    else:
        Xp = X

    # center
    Xc = Xp - Xp.mean(axis=0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T  # (N,2)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")


def plot_pareto_parallel_coords(F: np.ndarray, title: str, max_points: int = 300):
    """
    Common for multi-objective: parallel coordinate plot of objectives.
    """
    N, M = F.shape
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        Fp = F[idx]
    else:
        Fp = F

    # normalize each objective for visualization
    mins = Fp.min(axis=0)
    maxs = Fp.max(axis=0)
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    Fn = (Fp - mins) / denom

    plt.figure(figsize=(10, 4))
    xs = np.arange(M)
    for n in range(Fn.shape[0]):
        plt.plot(xs, Fn[n, :], alpha=0.2)
    plt.title(title)
    plt.xticks(xs, [f"obj{m}" for m in range(M)])
    plt.ylabel("normalized value")
