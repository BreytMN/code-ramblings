import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._generics import CountDistribution, CountParams

__all__ = ["CountMixtureModel"]


@dataclass
class MixtureParams:
    """Parameters for a two-component mixture model.

    Attributes:
        pi: Mixture weight for first component (1-pi for second component).
        dist1_params: Parameters for first distribution.
        dist2_params: Parameters for second distribution.
    """

    pi: float
    dist1_params: CountParams
    dist2_params: CountParams


@dataclass
class MixtureResults:
    """Results from fitting a two-component mixture model.

    Attributes:
        params: Fitted model parameters.
        log_likelihood: Final log-likelihood value.
        converged: Whether the EM algorithm converged.
        n_iterations: Number of iterations performed.
    """

    params: MixtureParams
    log_likelihood: float
    converged: bool
    n_iterations: int


class CountMixtureModel:
    """Generic two-component mixture model for count distributions using EM algorithm.

    This class implements a finite mixture model that can combine any two
    distributions implementing the CountDistribution protocol. The model uses
    the Expectation-Maximization (EM) algorithm for parameter estimation.
    Common applications include zero-inflated models and modeling heterogeneous
    count data populations.

    Examples:
        Create and fit a Poisson-NegativeBinomial mixture:

        >>> from code_ramblings.models.stats import NegativeBinomialDistribution, PoissonDistribution
        >>> import numpy as np
        >>>
        >>> poisson_dist = PoissonDistribution()
        >>> negbinom_dist = NegativeBinomialDistribution()
        >>> model = CountMixtureModel(poisson_dist, negbinom_dist)
        >>>
        >>> # Generate test data from two populations
        >>> np.random.seed(42)
        >>> poisson_data = np.random.poisson(3, 60)
        >>> negbinom_data = np.random.negative_binomial(5, 0.3, 40)
        >>> mixed_data = np.concatenate([poisson_data, negbinom_data])
        >>>
        >>> # Fit the model
        >>> results = model.fit(mixed_data)
        >>> results.converged
        True
        >>> (0.3 < results.params.pi < 0.8).item()  # Should recover approximate mixture weight
        True
    """

    def __init__(
        self,
        distribution1: CountDistribution,
        distribution2: CountDistribution,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """Initialize the mixture model.

        Args:
            distribution1: First distribution component.
            distribution2: Second distribution component.
            max_iter: Maximum number of EM iterations.
            tol: Convergence tolerance for log-likelihood change.

        Examples:
            >>> from code_ramblings.models.stats import DegenerateDistribution, PoissonDistribution
            >>> poisson = PoissonDistribution()
            >>> degenerate = DegenerateDistribution()
            >>> model = CountMixtureModel(degenerate, poisson, max_iter=500, tol=1e-8)
            >>> model.max_iter
            500
            >>> model.tol
            1e-08
        """
        self.distribution1 = distribution1
        self.distribution2 = distribution2
        self.max_iter = max_iter
        self.tol = tol

    def __repr__(self) -> str:
        return (
            "MixtureModel(\n"
            f"    {self.distribution1.__repr__()},\n"
            f"    {self.distribution2.__repr__()}\n"
            ")"
        )

    def fit(self, data: Sequence[int]) -> MixtureResults:
        """Fit the mixture model using Expectation-Maximization algorithm.

        The EM algorithm alternates between computing posterior probabilities
        (E-step) and updating parameters (M-step) until convergence.

        Args:
            data: Sequence of non-negative integer count data.

        Returns:
            MixtureResults object containing fitted parameters and diagnostics.

        Raises:
            ValueError: If data is empty or contains negative values.

        Examples:
            Fit a zero-inflated Poisson model (degenerate at 0 + Poisson):

            >>> from code_ramblings.models.stats import DegenerateDistribution, PoissonDistribution
            >>> import numpy as np
            >>> degenerate = DegenerateDistribution()
            >>> poisson = PoissonDistribution()
            >>> model = CountMixtureModel(degenerate, poisson)
            >>>
            >>> # Generate zero-inflated data
            >>> np.random.seed(123)
            >>> zip_data = np.concatenate([
            ...     np.zeros(40, dtype=int),  # Extra zeros from degenerate
            ...     np.random.poisson(2, 60)  # Regular Poisson
            ... ])
            >>> results = model.fit(zip_data)
            >>> results.converged
            True
            >>> results.params.dist1_params.point == 0.0  # Degenerate at 0
            True
        """
        data_array = np.array(data, dtype=int)

        if len(data_array) == 0:
            raise ValueError("Data cannot be empty")
        if np.any(data_array < 0):
            raise ValueError("Data must contain non-negative integers")

        pi = self._m_step(data_array)
        prev_log_likelihood = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(data_array, pi)
            pi = self._m_step(data_array, responsibilities)
            log_likelihood = self._log_likelihood(data_array, pi)

            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                converged = True
                break

            prev_log_likelihood = log_likelihood

        if not converged:
            msg = f"EM algorithm did not converge after {self.max_iter} iterations"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        self.params = MixtureParams(
            pi=pi,
            dist1_params=self.distribution1.params,
            dist2_params=self.distribution2.params,
        )

        return MixtureResults(
            params=self.params,
            log_likelihood=prev_log_likelihood,
            converged=converged,
            n_iterations=iteration + 1,
        )

    def predict_proba(self, data: Sequence[int]) -> tuple[NDArray, NDArray]:
        """Predict component probabilities for each observation.

        Computes the posterior probability that each observation belongs to
        each component of the mixture.

        Args:
            data: Sequence of count data.

        Returns:
            Tuple of (dist1_probabilities, dist2_probabilities) where each array
            contains the posterior probability of each observation belonging to
            the respective component.

        Raises:
            AttributeError: If model has not been fitted yet.

        Examples:
            >>> from code_ramblings.models.stats import NegativeBinomialDistribution, PoissonDistribution
            >>> import numpy as np
            >>> poisson = PoissonDistribution()
            >>> negbinom = NegativeBinomialDistribution()
            >>> model = CountMixtureModel(poisson, negbinom)
            >>>
            >>> np.random.seed(42)
            >>> train_data = np.concatenate([
            ...     np.random.poisson(2, 50),
            ...     np.random.negative_binomial(3, 0.3, 50)
            ... ])
            >>> results = model.fit(train_data)
            >>>
            >>> test_data = [0, 1, 5, 10]
            >>> dist1_probs, dist2_probs = model.predict_proba(test_data)
            >>> len(dist1_probs) == len(test_data)
            True
            >>> np.allclose(dist1_probs + dist2_probs, 1.0)
            True
        """
        if not hasattr(self, "params"):
            raise AttributeError("Model must be fitted before making predictions")

        data_array = np.array(data, dtype=int)
        responsibilities = self._e_step(data_array, self.params.pi)
        return responsibilities, 1 - responsibilities

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate samples from the fitted mixture distribution.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator for reproducible sampling.
            seed: Random seed (used only if rng is None).

        Returns:
            Array of generated count samples from the mixture distribution.

        Raises:
            AttributeError: If model has not been fitted yet.
            ValueError: If n_samples is not positive.

        Examples:
            >>> from code_ramblings.models.stats import NegativeBinomialDistribution, PoissonDistribution
            >>> import numpy as np
            >>> poisson = PoissonDistribution()
            >>> negbinom = NegativeBinomialDistribution()
            >>> model = CountMixtureModel(poisson, negbinom)
            >>>
            >>> np.random.seed(42)
            >>> train_data = np.concatenate([
            ...     np.random.poisson(3, 100),
            ...     np.random.negative_binomial(5, 0.4, 100)
            ... ])
            >>> results = model.fit(train_data)
            >>>
            >>> # Generate new samples
            >>> samples = model.sample(50, seed=123)
            >>> len(samples) == 50
            True
            >>> np.all(samples >= 0).item()  # All samples should be non-negative
            True
            >>> samples.dtype in [np.int32, np.int64]  # Returns integers
            True
        """
        if not hasattr(self, "params"):
            raise AttributeError("Model must be fitted before sampling")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if rng is None:
            rng = np.random.default_rng(seed)
        params = self.params

        component_choices = rng.binomial(1, params.pi, n_samples)

        dist1_samples = self.distribution1.sample(n_samples, rng=rng)
        dist2_samples = self.distribution2.sample(n_samples, rng=rng)

        return np.where(component_choices, dist1_samples, dist2_samples).astype(int)

    def _log_likelihood(self, data: NDArray, pi: float) -> float:
        """Compute log-likelihood of the mixture model.

        Args:
            data: Array of count data.
            pi: Current mixture weight for first component.

        Returns:
            Log-likelihood value of the data given the parameters.
        """
        dist1_likes = self.distribution1.pmf(data)
        dist2_likes = self.distribution2.pmf(data)

        mixture_likes = pi * dist1_likes + (1 - pi) * dist2_likes
        mixture_likes = np.maximum(mixture_likes, 1e-300)  # Prevent log(0)

        return np.log(mixture_likes).sum()

    def _m_step(self, data: NDArray, responsibilities: NDArray | None = None) -> float:
        """Maximization step: update parameters using current responsibilities.

        Args:
            data: Array of count data.
            responsibilities: Posterior probabilities from E-step. If None,
                uses equal weights (0.5 for each component).

        Returns:
            Updated mixture weight for first component.
        """
        if responsibilities is None:
            responsibilities = np.full_like(data, 0.5, dtype=np.float64)

        pi = np.mean(responsibilities)

        # Update distribution parameters using weighted updates
        self.distribution1.update_params(data, responsibilities)
        self.distribution2.update_params(data, 1 - responsibilities)

        return pi

    def _e_step(self, data: NDArray, pi: float) -> NDArray:
        """Expectation step: compute posterior probabilities (responsibilities).

        Args:
            data: Array of count data.
            pi: Current mixture weight for first component.

        Returns:
            Posterior probabilities (responsibilities) for first component.
        """
        # Component likelihoods
        dist1_likes = self.distribution1.pmf(data)
        dist2_likes = self.distribution2.pmf(data)

        # Avoid numerical issues
        dist1_likes = np.maximum(dist1_likes, 1e-300)
        dist2_likes = np.maximum(dist2_likes, 1e-300)

        # Posterior probabilities (responsibilities)
        numerator = pi * dist1_likes
        denominator = pi * dist1_likes + (1 - pi) * dist2_likes

        responsibilities = numerator / denominator
        return responsibilities
