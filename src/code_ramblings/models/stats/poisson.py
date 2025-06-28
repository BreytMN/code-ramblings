from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore

from ._generics import CountDistribution, CountParams


@dataclass
class PoissonParams(CountParams):
    """Parameters for Poisson distribution.

    Attributes:
        lambda_: Rate parameter (lambda > 0).
    """

    lambda_: float


class PoissonDistribution(CountDistribution[PoissonParams]):
    """Poisson distribution for modeling count data.

    The Poisson distribution models the number of events occurring in a fixed
    interval of time or space, given a known constant rate. It's commonly used
    for modeling count data where events occur independently.

    Examples:
        >>> import numpy as np
        >>> dist = PoissonDistribution(lambda_=3.0)
        >>> dist.lambda_
        3.0
    """

    def __init__(self, lambda_: float = 2.0) -> None:
        """Initialize Poisson distribution.

        Args:
            lambda_: Rate parameter (must be > 0).

        Examples:
            >>> dist = PoissonDistribution()
            >>> dist.lambda_
            2.0
            >>> dist2 = PoissonDistribution(lambda_=5.0)
            >>> dist2.lambda_
            5.0
        """
        self.params = PoissonParams(lambda_=lambda_)

    def __repr__(self) -> str:
        return f"PoissonDistribution(lambda_={round(self.params.lambda_, 2)})"

    @property
    def lambda_(self) -> float:
        """Rate parameter."""
        return self.params.lambda_

    @lambda_.setter
    def lambda_(self, v: float) -> None:
        self.params.lambda_ = float(v)

    def pmf(self, x: NDArray) -> NDArray:
        """Compute Poisson probability mass function.

        Args:
            x: Array of non-negative integers.

        Returns:
            Array of probability mass values.

        Examples:
            >>> import numpy as np
            >>> dist = PoissonDistribution(lambda_=2.0)
            >>> x = np.array([0, 1, 2, 3])
            >>> probs = dist.pmf(x)
            >>> len(probs) == len(x)
            True
            >>> np.all(probs >= 0).item()
            True
            >>> np.all(probs <= 1).item()
            True
        """
        return stats.poisson.pmf(x, self.params.lambda_)

    def update_params(self, data: NDArray, weights: NDArray | None = None) -> None:
        """Update Poisson parameter using weighted maximum likelihood estimation.

        The maximum likelihood estimator for the Poisson rate parameter is
        the weighted sample mean.

        Args:
            data: Array of count data.
            weights: Array of weights for each observation. If None, all
                observations are weighted equally.

        Examples:
            >>> import numpy as np
            >>> dist = PoissonDistribution()
            >>> data = np.array([1, 2, 3, 4])
            >>> old_lambda = dist.lambda_
            >>> dist.update_params(data)
            >>> # Parameter should be updated to sample mean
            >>> abs(dist.lambda_ - 2.5) < 1e-10
            True
        """
        if weights is None:
            weights = np.ones_like(data, dtype=float)

        weighted_sum = np.sum(weights)
        if weighted_sum > 0:
            self.lambda_ = np.sum(weights * data) / weighted_sum

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate samples from Poisson distribution.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator for reproducible sampling.
            seed: Random seed (used only if rng is None).

        Returns:
            Array of Poisson-distributed samples.

        Examples:
            >>> import numpy as np
            >>> dist = PoissonDistribution(lambda_=2.0)
            >>> samples = dist.sample(10, seed=42)
            >>> len(samples)
            10
            >>> np.all(samples >= 0).item()
            True
            >>> samples.dtype in [np.int32, np.int64]
            True
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        return rng.poisson(self.params.lambda_, n_samples)
