from dataclasses import dataclass
import numpy as np

from core.plotter import plot_error_heatmap, plot_error_distribution
from src.core.base_experiment import BaseExperiment


class SingleOperationExperiment(BaseExperiment):
    """
    Experiment simulating repeated floating-point accumulation (c = c + x)
    to measure relative error propagation.

    Compares low-precision accumulation against high-precision ground truth.

    Parameters
    ----------
    dtype_list : list of dtype
        List of floating-point precisions to evaluate (e.g., [np.float16, np.float32]).
    sizes : list of int
        List of input lengths (number of accumulation steps).
    repeats : int
        Number of random trials per size.
    value_range : tuple of float
        Range of values to sample from (uniform distribution).
    """

    def __init__(self, dtype_list, sizes, repeats, value_range):
        super().__init__(dtype_list, sizes, repeats)
        self.value_range = value_range

    def run_single_size(self, size, dtype):
        """
        Perform repeated accumulation with specified size and precision.

        Parameters
        ----------
        size : int
            Number of elements to accumulate.
        dtype : dtype
            Low precision data type to simulate (e.g., np.float32).

        Returns
        -------
        rel_err : float
            Relative error between low-precision and high-precision results.
        """
        x = np.random.uniform(self.value_range[0], self.value_range[1], size)
        low_x = x.astype(dtype)
        high_x = x.astype(np.float64)

        low_ret = dtype(0)
        high_ret = np.float64(0)
        for i in range(size):
            high_ret = low_ret + high_x[i]  # high-precision reference
            low_ret = low_ret + low_x[i]  # low-precision accumulation

        rel_err = (low_ret - high_ret) / high_ret if high_ret != 0 else 0
        return rel_err


if __name__ == "__main__":
    # Configuration parameters
    sizes = range(1, 1000)  # Varying accumulation lengths
    repeats = 100  # Number of random trials
    value_range = (0, 2)  # Range of input values
    dtype = np.float32  # Precision to test (e.g., np.float16, np.float32)

    # Initialize and run experiment
    exp = SingleOperationExperiment(
        dtype_list=[dtype],
        sizes=sizes,
        repeats=repeats,
        value_range=value_range
    )
    exp.run()
    # Extract results and visualize
    errors_matrix = np.array(exp.results[dtype])
    plot_error_heatmap(errors_matrix, title="Heatmap of Float32 Error")
    plot_error_distribution(errors_matrix, title="Distribution of Float32 Error")
