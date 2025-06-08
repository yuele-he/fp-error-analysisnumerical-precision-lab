"""
simulate_relative_errors.py

This script simulates the propagation of relative errors in single floating-point addition operations (c = c + x).
Core Idea:
----------
For each step, we compare:
- low_ret += low_x[i]: actual low-precision computation (e.g., float32), subject to rounding
- high_ret = low_ret + high_x[i]: a high-precision reference computed using float64

This allows us to observe how rounding errors accumulate depending on the value magnitude and accumulation direction.
We repeat the simulation using multiple random sequences to capture the distribution of step-wise relative errors.

The results are visualized using:
- A 2D heatmap of relative error vs step index (plt.hist2d)
- A histogram and kernel density estimate (KDE) of error values (sns.histplot)

This code is designed to reproduce and extend Figure 3(a) and Figure 3(b) in the thesis.

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_relative_errors(n_steps=10 ** 3, n_trials=100, value_range=(0, 2),
                             low_precision=np.float32, high_precision=np.float64):
    """
    simulate relative errors during repeated accumulation (c = c + x) of random values.

    Parameters:
        n_steps (int): Number of accumulation steps per trial.
        n_trials (int): Number of random sequences (independent trials).
        value_range (tuple): Range from which random values are sampled.
        low_precision (dtype): Floating-point type used to simulate low-precision computation.
        high_precision (dtype): Floating-point type used for ground truth.

    Returns:
        errors_matrix (ndarray): Array of shape (n_trials, n_steps) containing relative errors.
    """
    errors_matrix = np.zeros((n_trials, n_steps))

    for trial in range(n_trials):

        x = np.random.uniform(value_range[0], value_range[1], n_steps)
        low_x = x.astype(low_precision)
        high_x = x.astype(high_precision)

        low_ret = low_precision(0)
        high_ret = high_precision(0)
        for i in range(n_steps):
            high_ret = low_ret + high_x[i]
            low_ret = low_ret + low_x[i]

            errors = (low_ret - high_ret) / high_ret
            errors_matrix[trial][i] = errors

    return errors_matrix


def plot_figure3(errors_matrix, save_path=None, title_prefix=""):
    """
    Plot dual-panel visualization similar to Figure 3 in the thesis.

    Left: 2D histogram (hist2d) of error over accumulation steps.
    Right: Histogram + KDE of error distribution (aligned vertically).

    Parameters:
        errors_matrix (ndarray): (n_trials x n_steps) array of relative errors.
        save_path (str): If provided, save the figure to this path.
        title_prefix (str): Title prefix for both subplots.
    """
    n_trials, n_steps = errors_matrix.shape
    flat_errors = errors_matrix.flatten()
    step_indices = np.tile(np.arange(n_steps), n_trials)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [4, 1]})

    # 左图：误差 vs 步骤，热度图
    hb = ax1.hist2d(step_indices, flat_errors, bins=(128, 128), cmap='coolwarm', cmin=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Relative Error")
    ax1.set_title(f"{title_prefix} Error Heatmap over Steps")

    # Add vertical dotted lines at positions 2^n
    for n in range(5, int(np.log2(n_steps)) + 1):
        ax1.axvline(2 ** n, color='red', linestyle='dotted', linewidth=1)

    fig.colorbar(hb[3], ax=ax1, label='Count')

    # --- Right: Histogram + KDE of relative error (horizontal) ---
    sns.histplot(y=flat_errors, bins=80, kde=True, ax=ax2, color='skyblue', orientation="horizontal")
    ax2.set_ylabel("Relative Error")
    ax2.set_xlabel("Density")
    ax2.set_title(f"{title_prefix} Error Distribution")
    # ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# Simulate Figure 3(a): Values in [0, 2], non-zero mean
errors_a = simulate_relative_errors(value_range=(0, 2))
plot_figure3(errors_a, title_prefix="Element in [0, 2]")

# Simulate Figure 3(b): Values in [-1, 1], mean ≈ 0
errors_b = simulate_relative_errors(value_range=(-1, 1))
plot_figure3(errors_b, title_prefix="Mean ≈ 0")
