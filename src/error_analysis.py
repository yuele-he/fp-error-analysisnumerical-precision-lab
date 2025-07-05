import pprint

import numpy as np
import matplotlib.pyplot as plt
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

    rhs = Q.T @ b
    x = np.linalg.lstsq(R, rhs, rcond=None)[0]
    ls = np.linalg.norm(R @ x - Q.T @ b) / np.linalg.norm(b)
    return recon, orth, ls


# Experimental setup
methods = ['householder', 'wy', 'block']
precisions = [np.float32, np.float64]
precision_labels = ['float32', 'float64']
metrics = ['Reconstruction', 'Orthogonality', 'Least Squares']
values = np.logspace(np.log10(10), np.log10(512), num=20)
matrix_size = np.round(values).astype(int)
# n_values = [int(1.5 ** i) for i in range(8, 15)]  # n = 16 to 1024
n_trials = 50

# Storage for plotting and fitting
results = {method: {prec: {metric: [] for metric in metrics} for prec in precision_labels} for method in methods}
fit_params = {method: {prec: {} for prec in precision_labels} for method in methods}

# Run experiments
for method in methods:
    qr_func = qr_methods[method]
    for prec, label in zip(precisions, precision_labels):
        for n in matrix_size:
            m = 2 * n
            errors = {metric: [] for metric in metrics}
            for _ in range(n_trials):
                A = np.random.uniform(1, 2, size=(m, n)).astype(prec)
                A = A / np.linalg.norm(A)

                b = np.random.uniform(1, 2, size=(m, 1)).astype(prec)
                # try:
                recon, orth, ls = compute_qr_errors(A, b, qr_func)
                # print(recon, orth, ls)

                errors['Reconstruction'].append(recon)
                errors['Orthogonality'].append(orth)
                errors['Least Squares'].append(ls)
                # except:
                #     continue
            for metric in metrics:
                results[method][label][metric].append(errors[metric])
        # Fit log-log line
        for metric in metrics:
            y = np.array(np.max(results[method][label][metric], axis=1))
            x = np.array(matrix_size)
            logx = np.log(x)
            logy = np.log(y + 1e-20)
            k, c, *_ = linregress(logx, logy)
            fit_params[method][label][metric] = (k, c)
        print(method, label)
# Plot 3x2 grid
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i, metric in enumerate(metrics):
    for j, label in enumerate(precision_labels):
        ax = axes[i, j]
        for method, color in zip(methods, ['orange', 'royalblue', 'black']):
            y = results[method][label][metric]

            ax.scatter(np.repeat(matrix_size, n_trials), y, label=method, alpha=0.3, color=color)
            # Plot fitted line
            k, c = fit_params[method][label][metric]
            fit_line = np.exp(k * np.log(matrix_size) + c)
            ax.plot(matrix_size, fit_line, linestyle='--', color=color)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if i == 0:
            ax.set_title(f'{label} QR')
        if j == 0:
            ax.set_ylabel(f'{metric} Error')
        if i == 2:
            ax.set_xlabel('Matrix Width n')
        ax.grid(True, which='both', linestyle=':')
        if i == 0 and j == 1:
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
# 设置显示的最大行数和列数
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
print(pprint.pformat(fit_table_pivot))
# print(fit_table_pivot)
# fit_table_pivot
