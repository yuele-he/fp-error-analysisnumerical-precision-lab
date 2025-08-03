# src/core/plotter.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_error_heatmap(
    errors_matrix: np.ndarray,
    title: str = "Relative Error Heatmap",
    save_path: str = None,
    bins: int = 128,
):
    """
    绘制 step-wise 误差热力图。

    参数:
        errors_matrix (ndarray): (n_trials, n_steps) 误差矩阵。
        title (str): 图标题。
        save_path (str): 若提供路径则保存图像。
        bins (int): bin 数量。
    """
    n_trials, n_steps = errors_matrix.shape
    print("n_trials, n_steps", n_trials, n_steps)
    flat_errors = errors_matrix.flatten()
    step_indices = np.tile(np.arange(n_steps), n_trials)

    fig, ax = plt.subplots(figsize=(8, 5))

    hb = ax.hist2d(step_indices, flat_errors, bins=(bins, bins), cmap='coolwarm', cmin=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative Error")
    ax.set_title(title)

    # Add log2 reference lines
    for n in range(5, int(np.log2(n_steps)) + 1):
        ax.axvline(2 ** n, color='red', linestyle='dotted', linewidth=1)

    fig.colorbar(hb[3], ax=ax, label="Count")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def plot_error_distribution(
    errors_matrix: np.ndarray,
    title: str = "Relative Error Distribution",
    save_path: str = None,
    bins: int = 80,
):
    """
    绘制误差分布图（带 KDE 的直方图）。

    参数:
        errors_matrix (ndarray): (n_trials, n_steps) 误差矩阵。
        title (str): 图标题。
        save_path (str): 若提供路径则保存图像。
        bins (int): bin 数量。
    """
    flat_errors = errors_matrix.flatten()

    fig, ax = plt.subplots(figsize=(4, 5))
    sns.histplot(y=flat_errors, bins=bins, kde=True, ax=ax, color='skyblue', orientation="horizontal")

    ax.set_ylabel("Relative Error")
    ax.set_xlabel("Density")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
