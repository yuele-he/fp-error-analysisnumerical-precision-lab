import numpy as np
from abc import ABC, abstractmethod


class BaseExperiment(ABC):
    """
    Abstract base class for numerical error experiments.

    This class provides a framework to conduct repeated experiments across
    different data types and problem sizes. Derived classes should implement
    the `run_single_size` method to define the core numerical operation.

    Parameters
    ----------
    dtype_list : list of dtype or str
        List of floating-point precisions to be tested, e.g., [np.float16, np.float32].
    sizes : list of int
        List of input sizes (e.g., vector lengths or matrix dimensions).
    repeats : int
        Number of repeated trials per data type and size.
    """

    def __init__(self, dtype_list, sizes, repeats):
        self.dtype_list = dtype_list
        self.sizes = sizes
        self.repeats = repeats
        self.results = {}

    def run_single_trial(self, dtype):
        """
        Run a single trial across all problem sizes for given precision.

        Parameters
        ----------
        dtype : dtype
            The floating-point precision for this trial.

        Returns
        -------
        errors : list of float
            List of error values for each problem size in the current trial.
        """
        errors = []
        for size in self.sizes:
            error = self.run_single_size(size, dtype)
            errors.append(error)
        return errors

    @abstractmethod
    def run_single_size(self, size, dtype):
        """
        Abstract method: run a single experiment for a given size and precision.

        This method must be implemented by subclasses.

        Parameters
        ----------
        size : int
            The size of the input (e.g., vector length or matrix dimension).
        dtype : dtype
            Floating-point precision to use.

        Returns
        -------
        error : float
            Computed error metric (e.g., relative error).
        """
        pass

    def run(self):
        """
        Run the full experiment over all precisions and repeated trials.

        Fills the `self.results` dictionary with nested lists:
        results[dtype] = list of trial results, each a list of errors per size.
        """
        for dtype in self.dtype_list:
            self.results[dtype] = []
            for repeat in range(self.repeats):
                error = self.run_single_trial(dtype)
                self.results[dtype].append(error)
