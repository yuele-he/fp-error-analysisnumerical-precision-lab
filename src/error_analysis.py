import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

from qr_factorization import householder_qr, qr_wy, block_qr

# Mapping method names to functions
qr_methods = {
    'householder': householder_qr,
    'wy': qr_wy,
    'block': block_qr,
}

def compute_qr_errors(A, b, qr_func):
    Q, R = qr_func(A)
    recon = np.linalg.norm(A - Q @ R, 'fro') / np.linalg.norm(A, 'fro')
    orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])) / np.linalg.norm(Q)
    x = np.linalg.solve(R, Q.T @ b)
    ls = np.linalg.norm(R @ x - Q.T @ b) / np.linalg.norm(b)
    return recon, orth, ls

# Experimental setup
methods = ['householder', 'wy', 'block']
precisions = [np.float16, np.float32, np.float64]
precision_labels = ['float16', 'float32', 'float64']
metrics = ['Reconstruction', 'Orthogonality', 'Least Squares']
# TODO 4,11
n_values = [2**i for i in range(4, 6)]  # n = 16 to 1024
n_trials = 2

# Storage for plotting and fitting
results = {method: {prec: {metric: [] for metric in metrics} for prec in precision_labels} for method in methods}
fit_params = {method: {prec: {} for prec in precision_labels} for method in methods}

# Run experiments
for method in methods:
    qr_func = qr_methods[method]
    for prec, label in zip(precisions, precision_labels):
        for n in n_values:
            m = 2 * n
            max_errors = {metric: 0.0 for metric in metrics}
            for _ in range(n_trials):
                A = np.random.randn(m, n).astype(prec)
                b = np.random.randn(m).astype(prec)
                try:
                    recon, orth, ls = compute_qr_errors(A, b, qr_func)
                    max_errors['Reconstruction'] = max(max_errors['Reconstruction'], recon)
                    max_errors['Orthogonality'] = max(max_errors['Orthogonality'], orth)
                    max_errors['Least Squares'] = max(max_errors['Least Squares'], ls)
                except:
                    continue
            for metric in metrics:
                results[method][label][metric].append(max_errors[metric])
        # Fit log-log line
        for metric in metrics:
            y = np.array(results[method][label][metric])
            x = np.array(n_values)
            logx = np.log(x)
            logy = np.log(y + 1e-20)
            k, c, *_ = linregress(logx, logy)
            fit_params[method][label][metric] = (k, c)

# Plot 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for i, metric in enumerate(metrics):
    for j, method in enumerate(methods):
        ax = axes[i, j]
        for label, color in zip(precision_labels, ['orange', 'royalblue', 'black']):
            y = results[method][label][metric]
            ax.plot(n_values, y, label=label, marker='o', color=color)
            # Plot fitted line
            k, c = fit_params[method][label][metric]
            fit_line = np.exp(k * np.log(n_values) + c)
            ax.plot(n_values, fit_line, linestyle='--', color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i == 0:
            ax.set_title(f'{method.capitalize()} QR')
        if j == 0:
            ax.set_ylabel(f'{metric} Error')
        if i == 2:
            ax.set_xlabel('Matrix Width n')
        ax.grid(True, which='both', linestyle=':')
        if i == 0 and j == 2:
            ax.legend(loc='upper left')

plt.tight_layout()
plt.show()

# Prepare fit parameter table
import pandas as pd

rows = []
for method in methods:
    for prec in precision_labels:
        for metric in metrics:
            k, c = fit_params[method][prec][metric]
            rows.append({
                "Method": method,
                "Precision": prec,
                "Metric": metric,
                "k": round(k, 3),
                "c": round(c, 3)
            })

fit_table = pd.DataFrame(rows)
fit_table_pivot = fit_table.pivot(index=["Method", "Precision"], columns="Metric", values=["k", "c"])
fit_table_pivot.reset_index(inplace=True)
# fit_table_pivot