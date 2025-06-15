import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray
from scipy import stats  # type: ignore


def compute_value_frequencies(distributions: dict[str, NDArray]) -> pd.DataFrame:
    """
    Compute value frequencies for multiple named distributions.

    Takes a dictionary containing one or more named distributions (arrays), and
    returns a DataFrame with the frequency count of each value for each distribution.
    Uses numpy's optimized unique function for efficient computation before
    converting to pandas format.

    Args:
        distributions: Dictionary where keys are distribution names and values
            are numpy arrays containing the distribution samples.

    Returns:
        DataFrame with columns:
            - 'distribution': Name of the distribution
            - 'value': The observed value
            - 'count': Frequency of that value in the distribution

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> poisson_data = rng.poisson(2, 100)
        >>> normal_data = rng.normal(0, 1, 100).astype(int)  # Convert to int for counting
        >>> distributions = {
        ...     "poisson": poisson_data[:10],  # Use smaller sample for doctest
        ...     "normal": normal_data[:10],
        ... }
        >>> result = compute_value_frequencies(distributions)
        >>> isinstance(result, pd.DataFrame)
        True
        >>> set(result.columns) == {"distribution", "value", "count"}
        True
        >>> result["distribution"].nunique()
        2
        >>> all(result["count"] > 0)  # All counts should be positive
        True
    """

    result_data: list[pd.DataFrame] = []

    for dist_name, dist_array in distributions.items():
        data = dict(zip(["value", "count"], np.unique(dist_array, return_counts=True)))
        result_data.append(pd.DataFrame(data).assign(distribution=dist_name))

    return pd.concat(result_data, ignore_index=True)


def poisson_pmf(x: NDArray, lam: float) -> NDArray:
    """Compute Poisson probability mass function.

    Args:
        x: Array of non-negative integers.
        lam: Rate parameter (lambda > 0).

    Returns:
        Array of probability mass values.

    Examples:
        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3])
        >>> probs = poisson_pmf(x, 2.0)
        >>> np.allclose(probs, [0.135335, 0.270671, 0.270671, 0.180447], atol=1e-5)
        True
    """
    return stats.poisson.pmf(x, lam)


def negbinom_pmf(x: NDArray, r: float, p: float) -> NDArray:
    """Compute negative binomial probability mass function.

    Args:
        x: Array of non-negative integers (number of failures).
        r: Number of successes parameter (r > 0).
        p: Success probability parameter (0 < p < 1).

    Returns:
        Array of probability mass values.

    Examples:
        >>> import numpy as np
        >>> x = np.array([0, 1, 2, 3])
        >>> probs = negbinom_pmf(x, 5.0, 0.5)
        >>> np.allclose(probs, [0.03125, 0.078125, 0.1171875, 0.1367188], atol=1e-6)
        True
    """
    return stats.nbinom.pmf(x, r, p)


@dataclass
class MixtureParams:
    """Parameters for a Poisson-Negative Binomial mixture model.

    Attributes:
        pi: Mixture weight for Poisson component (1-pi for NegBinom).
        poisson_lambda: Rate parameter for Poisson component.
        negbinom_r: Number of failures parameter for Negative Binomial.
        negbinom_p: Success probability for Negative Binomial.
    """

    pi: float
    poisson_lambda: float
    negbinom_r: float
    negbinom_p: float


@dataclass
class MixtureResults:
    """Results from fitting a Poisson-Negative Binomial mixture model.

    Attributes:
        params: Fitted model parameters.
        log_likelihood: Final log-likelihood value.
        converged: Whether the algorithm converged.
        n_iterations: Number of iterations performed.
    """

    params: MixtureParams
    log_likelihood: float
    converged: bool
    n_iterations: int


class PoissonNegBinomMixture:
    """Poisson-Negative Binomial mixture model using EM algorithm.

    Fits a mixture of Poisson and Negative Binomial distributions to count data
    using the Expectation-Maximization algorithm.

    Examples:
        >>> model = PoissonNegBinomMixture()
        >>> # Generate mixed count data
        >>> np.random.seed(42)
        >>> poisson_data = np.random.poisson(3, 60)
        >>> negbinom_data = np.random.negative_binomial(5, 0.3, 40)
        >>> mixed_data = np.concatenate([poisson_data, negbinom_data])
        >>> results = model.fit(mixed_data)
        >>> 0.4 < results.params.pi < 0.8  # Should recover mixture weight
        np.True_
        >>> results.converged
        True
    """

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """Initialize the mixture model.

        Args:
            max_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for log-likelihood.

        Examples:
            >>> model = PoissonNegBinomMixture(max_iter=500, tol=1e-8)
            >>> model.max_iter
            500
            >>> model.tol
            1e-08
        """
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data: Sequence[int]) -> MixtureResults:
        """Fit the mixture model using EM algorithm.

        Args:
            data: Sequence of non-negative integer count data.

        Returns:
            MixtureResults object containing fitted parameters and diagnostics.

        Raises:
            ValueError: If data is empty or contains negative values.

        Examples:
            >>> model = PoissonNegBinomMixture()
            >>> # Simulate pure Poisson data
            >>> np.random.seed(123)
            >>> poisson_data = np.random.poisson(2, 100)
            >>> results = model.fit(poisson_data)
            >>> results.converged
            True
            >>> 1.5 < results.params.poisson_lambda < 2.5  # Should be close to 2
            np.True_
        """
        data_array = np.array(data, dtype=int)

        if len(data_array) == 0:
            raise ValueError("Data cannot be empty")
        if np.any(data_array < 0):
            raise ValueError("Data must contain non-negative integers")

        mixture_params = self.initialize_parameters(data_array)

        prev_log_likelihood = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            responsibilities = self.e_step(data_array, mixture_params)
            mixture_params = self.m_step(data_array, responsibilities)
            log_likelihood = self.log_likelihood(data_array, mixture_params)

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                converged = True
                break

            prev_log_likelihood = log_likelihood

        if not converged:
            msg = f"EM algorithm did not converge after {self.max_iter} iterations"
            warnings.warn(msg)

        self.params = mixture_params

        return MixtureResults(
            params=self.params,
            log_likelihood=prev_log_likelihood,
            converged=converged,
            n_iterations=iteration + 1,
        )

    def predict_proba(self, data: Sequence[int]) -> tuple[NDArray, NDArray]:
        """Predict component probabilities for each observation.

        Args:
            data: Sequence of count data.

        Returns:
            Tuple of (poisson_probabilities, negbinom_probabilities).

        Raises:
            AttributeError: If model has not been fitted yet.

        Examples:
            >>> model = PoissonNegBinomMixture()
            >>> np.random.seed(42)
            >>> train_data = np.concatenate([
            ...     np.random.poisson(2, 50),
            ...     np.random.negative_binomial(3, 0.3, 50)
            ... ])
            >>> results = model.fit(train_data)
            >>> test_data = [0, 1, 5, 10]
            >>> poisson_probs, negbinom_probs = model.predict_proba(test_data)
            >>> len(poisson_probs) == len(test_data)
            True
            >>> np.allclose(poisson_probs + negbinom_probs, 1.0)
            True
        """
        if not hasattr(self, "params"):
            raise AttributeError("Model must be fitted before making predictions")

        data_array = np.array(data, dtype=int)
        responsibilities = self.e_step(data_array, self.params)
        return responsibilities, 1 - responsibilities

    def sample(self, n_samples: int, random_state: int | None = None) -> NDArray:
        """Generate samples from the fitted mixture distribution.

        Args:
            n_samples: Number of samples to generate.
            random_state: Random seed for reproducible sampling.

        Returns:
            Array of generated count samples.

        Raises:
            AttributeError: If model has not been fitted yet.
            ValueError: If n_samples is not positive.

        Examples:
            >>> model = PoissonNegBinomMixture()
            >>> np.random.seed(42)
            >>> # Fit to known mixture
            >>> train_data = np.concatenate([
            ...     np.random.poisson(3, 100),
            ...     np.random.negative_binomial(5, 0.4, 100)
            ... ])
            >>> results = model.fit(train_data)
            >>> # Generate new samples
            >>> samples = model.sample(50)
            >>> len(samples) == 50
            True
            >>> np.all(samples >= 0)  # All samples should be non-negative
            np.True_
            >>> isinstance(samples[0], (int, np.integer))  # Should be integers
            True
        """
        if not hasattr(self, "params"):
            raise AttributeError("Model must be fitted before sampling")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(random_state)
        params = self.params

        return np.where(
            rng.binomial(1, params.pi, n_samples),
            rng.poisson(params.poisson_lambda, n_samples),
            rng.negative_binomial(params.negbinom_r, params.negbinom_p, n_samples),
        ).astype(int)

    @staticmethod
    def log_likelihood(data: NDArray, mixture_params: MixtureParams) -> float:
        """Compute log-likelihood of the mixture model.

        Args:
            data: Array of count data.
            mixture_params: Model parameters.

        Returns:
            Log-likelihood value.

        Examples:
            >>> data = np.array([0, 1, 2, 3])
            >>> params = MixtureParams(pi=0.5, poisson_lambda=2.0,
            ...                       negbinom_r=5.0, negbinom_p=0.4)
            >>> ll = PoissonNegBinomMixture.log_likelihood(data, params)
            >>> isinstance(ll, float)
            True
            >>> ll < 0  # Log-likelihood should be negative
            np.True_
        """
        poisson_likes = poisson_pmf(data, mixture_params.poisson_lambda)
        negbinom_likes = negbinom_pmf(
            data,
            mixture_params.negbinom_r,
            mixture_params.negbinom_p,
        )

        mixture_likes = (
            mixture_params.pi * poisson_likes + (1 - mixture_params.pi) * negbinom_likes
        )
        mixture_likes = np.maximum(mixture_likes, 1e-300)

        return np.log(mixture_likes).sum()

    @staticmethod
    def initialize_parameters(data: NDArray) -> MixtureParams:
        """Initialize parameters using method of moments and heuristics.

        Args:
            data: Array of count data.

        Returns:
            Initial parameter estimates.

        Examples:
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> params = PoissonNegBinomMixture.initialize_parameters(data)
            >>> 0.0 < params.pi < 1.0
            True
            >>> params.poisson_lambda > 0
            np.True_
            >>> params.negbinom_r > 0
            True
            >>> 0.0 < params.negbinom_p < 1.0
            True
        """
        mean_data = np.mean(data)
        var_data = np.var(data)

        if var_data > mean_data * 1.5:
            pi = 0.3  # Less Poisson component
        else:
            pi = 0.7  # More Poisson component

        # Initialize Poisson lambda as sample mean
        lambda_ = mean_data

        # Initialize negative binomial parameters
        # Using method of moments: mean = r(1-p)/p, var = r(1-p)/p^2
        if var_data > mean_data:
            p = mean_data / var_data
            p = max(0.01, min(0.99, p))  # Keep p in valid range
            r = mean_data * p / (1 - p)
            r = max(0.1, r)  # Keep r positive
        else:
            p = 0.5
            r = 2.0

        return MixtureParams(
            pi=pi,
            poisson_lambda=lambda_,
            negbinom_r=r,
            negbinom_p=p,
        )

    @staticmethod
    def m_step(data: NDArray, responsibilities: NDArray) -> MixtureParams:
        """Maximization step: update parameters.

        Args:
            data: Array of count data.
            responsibilities: Posterior probabilities from E-step.

        Returns:
            Updated parameter estimates.

        Examples:
            >>> data = np.array([1, 2, 3, 4])
            >>> responsibilities = np.array([0.8, 0.6, 0.4, 0.2])
            >>> params = PoissonNegBinomMixture.m_step(data, responsibilities)
            >>> 0.0 < params.pi < 1.0
            np.True_
            >>> params.poisson_lambda > 0
            np.True_
        """
        # Update mixture weight
        pi = np.mean(responsibilities)

        # Update Poisson parameter (weighted MLE)
        poisson_lambda = np.sum(responsibilities * data) / np.sum(responsibilities)
        poisson_lambda = max(0.001, poisson_lambda)  # Keep positive

        # Update Negative Binomial parameters (approximate)
        negbinom_weights = 1 - responsibilities
        weighted_sum = np.sum(negbinom_weights)

        if weighted_sum > 0:
            weighted_mean = np.sum(negbinom_weights * data) / weighted_sum
            weighted_var = (
                np.sum(negbinom_weights * (data - weighted_mean) ** 2) / weighted_sum
            )

            # Method of moments for negative binomial
            if weighted_var > weighted_mean:
                negbinom_p = weighted_mean / weighted_var
                negbinom_p = max(0.01, min(0.99, negbinom_p))
                negbinom_r = weighted_mean * negbinom_p / (1 - negbinom_p)
                negbinom_r = max(0.1, negbinom_r)
            else:
                negbinom_p = 0.5
                negbinom_r = 2.0
        else:
            negbinom_p = 0.5
            negbinom_r = 2.0

        return MixtureParams(
            pi=pi,
            poisson_lambda=poisson_lambda,
            negbinom_r=negbinom_r,
            negbinom_p=negbinom_p,
        )

    @staticmethod
    def e_step(data: NDArray, mixture_params: MixtureParams) -> NDArray:
        """Expectation step: compute posterior probabilities.

        Args:
            data: Array of count data.
            mixture_params: Current parameter estimates.

        Returns:
            Posterior probabilities (responsibilities) for Poisson component.

        Examples:
            >>> data = np.array([0, 1, 2])
            >>> params = MixtureParams(pi=0.6, poisson_lambda=1.0,
            ...                       negbinom_r=2.0, negbinom_p=0.5)
            >>> responsibilities = PoissonNegBinomMixture.e_step(data, params)
            >>> len(responsibilities) == len(data)
            True
            >>> np.all((responsibilities >= 0) & (responsibilities <= 1))
            np.True_
        """
        pi = mixture_params.pi
        poisson_lambda = mixture_params.poisson_lambda
        negbinom_r = mixture_params.negbinom_r
        negbinom_p = mixture_params.negbinom_p

        # Component likelihoods
        poisson_likes = poisson_pmf(data, poisson_lambda)
        negbinom_likes = negbinom_pmf(data, negbinom_r, negbinom_p)

        # Avoid numerical issues
        poisson_likes = np.maximum(poisson_likes, 1e-300)
        negbinom_likes = np.maximum(negbinom_likes, 1e-300)

        # Posterior probabilities (responsibilities)
        numerator = pi * poisson_likes
        denominator = pi * poisson_likes + (1 - pi) * negbinom_likes
        denominator = np.maximum(denominator, 1e-300)

        responsibilities = numerator / denominator
        return responsibilities
