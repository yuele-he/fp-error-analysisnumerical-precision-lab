import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from mpmath import mp

mp.dps = 50  # decimal digits precision (≈ float168)


def compute_reference_inner_product(a, b):
    return sum(mp.mpf(x) * mp.mpf(y) for x, y in zip(a, b))


def simulate_inner_product_errors(prec_type=np.float32, trials=100, seed=40,
                                  value_range=(0, 2)):
    """
    Simulate relative errors in inner product computation under a given precision.

    Parameters:
        prec_type: The target low precision (e.g., np.float16 or np.float32)

        trials: Number of repetitions per vector length
        value_range: Range of random vector elements

    Returns:
        n_list: list of vector lengths
        max_errors: list of maximum relative errors for each n
        all_errors: raw errors for scatter plot
    """
    np.random.seed(seed)
    n_list = [2 ** i for i in range(4, 17)]  # 16 to 131,072
    max_errors = []
    all_errors = []

    for n in n_list:
        trial_errors = []

        for _ in range(trials):
            a = np.random.uniform(*value_range, size=n)
            b = np.random.uniform(*value_range, size=n)

            a_prec = a.astype(prec_type)
            b_prec = b.astype(prec_type)

            # True reference value in float64
            if prec_type in (np.float16, np.float32):
                reference_type = np.float64
                exact = np.dot(a.astype(reference_type), b.astype(reference_type))
            else:
                exact = compute_reference_inner_product(a, b)

            # Approximate inner product in low precision
            approx = np.dot(a_prec, b_prec)

            rel_error = abs(approx - exact) / (abs(exact))
            trial_errors.append(rel_error)

        max_errors.append(max(trial_errors))
        all_errors.append(trial_errors)

    return n_list, max_errors, all_errors


def fit_log_error_bound(n_list, max_errors):
    """
    Fit a linear model: log(error) = k * log(n) + c

    Returns:
        slope k, intercept c
    """
    log_n = np.log(n_list)
    log_e = np.log(max_errors)
    slope, intercept, _, _, _ = linregress(log_n, log_e)
    return slope, intercept


def plot_inner_product_errors(
        n_list,
        all_errors,
        k,
        c,
        eps_val,
        label,
        color,
        show_fit=True,
        ax=None
):
    """
    Plot error scatter and upper-bound fit to a given matplotlib Axes.
    If ax is None, create a new plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for i, n in enumerate(n_list):
        ax.scatter(
            [n] * len(all_errors[i]),
            [float(e) for e in all_errors[i]],
            alpha=0.3,
            color=color,
            s=20,
            label=f"{label} samples" if i == 0 else None  # only show label once
        )

    # Fit line
    n_vals = np.array(n_list)
    fit_y = np.exp(k * np.log(n_vals) + c) * 1.2  # lift slightly above points
    if show_fit:
        ax.plot(n_vals, fit_y, linestyle='--', color=color, label=f"{label} fit (k={k:.2f})")

    ax.axhline(eps_val, color=color, linestyle=':', linewidth=1, label=f"{label} ε ≈ {eps_val:.1e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Vector Length (log scale)")
    ax.set_ylabel("Relative Error (log scale)")
    ax.grid(True, which="both", linestyle="dotted", alpha=0.5)


if __name__ == "__main__":

    for value_range in [(-1, 1), (0, 2)]:
        fig, ax = plt.subplots(figsize=(8, 6))

        for precision, color in [(np.float32, "royalblue"), (np.float64, "darkorange")]:
            label = f"{precision.__name__}"

            n_list, max_errors, all_errors = simulate_inner_product_errors(
                prec_type=precision,
                value_range=value_range
            )

            k, c = fit_log_error_bound(n_list, [float(e) for e in max_errors])

            plot_inner_product_errors(
                n_list, all_errors, k, c,
                eps_val=np.finfo(precision).eps,
                label=label,
                color=color,
                ax=ax
            )

        title = f"Inner Product Error, value_range = {value_range}"
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()
