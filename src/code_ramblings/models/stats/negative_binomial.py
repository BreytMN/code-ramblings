from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats  # type: ignore

from ._generics import CountDistribution, CountParams


@dataclass
class NegativeBinomialParams(CountParams):
    """Parameters for negative binomial distribution.

    Attributes:
        r: Number of successes parameter (r > 0).
        p: Success probability parameter (0 < p < 1).
    """

    r: float
    p: float


class NegativeBinomialDistribution(CountDistribution[NegativeBinomialParams]):
    """Negative binomial distribution for modeling overdispersed count data.

    The negative binomial distribution models the number of failures before
    achieving r successes, with each trial having success probability p.
    It's commonly used for count data that exhibits more variance than would
    be expected under a Poisson model.

    Examples:
        >>> import numpy as np
        >>> dist = NegativeBinomialDistribution(r=5.0, p=0.3)
        >>> dist.r
        5.0
        >>> dist.p
        0.3
    """

    def __init__(self, r: float = 2.0, p: float = 0.5) -> None:
        """Initialize negative binomial distribution.

        Args:
            r: Number of successes parameter (must be > 0).
            p: Success probability parameter (must be 0 < p < 1).

        Examples:
            >>> dist = NegativeBinomialDistribution()
            >>> dist.r
            2.0
            >>> dist.p
            0.5
        """
        self.params = NegativeBinomialParams(r=float(r), p=float(p))

    def __repr__(self) -> str:
        return f"NegativeBinomialDistribution(r={round(self.params.r, 2)}, p={round(self.params.p, 2)})"

    @property
    def p(self) -> float:
        """Success probability parameter."""
        return self.params.p

    @p.setter
    def p(self, v: float) -> None:
        self.params.p = float(v)

    @property
    def r(self) -> float:
        """Number of successes parameter."""
        return self.params.r

    @r.setter
    def r(self, v: float) -> None:
        self.params.r = float(v)

    def pmf(self, x: NDArray) -> NDArray:
        """Compute negative binomial probability mass function.

        Args:
            x: Array of non-negative integers (number of failures).

        Returns:
            Array of probability mass values.

        Examples:
            >>> import numpy as np
            >>> dist = NegativeBinomialDistribution(r=2.0, p=0.5)
            >>> x = np.array([0, 1, 2, 3])
            >>> probs = dist.pmf(x)
            >>> len(probs) == len(x)
            True
            >>> np.all(probs >= 0).item()
            True
            >>> np.all(probs <= 1).item()
            True
        """
        return stats.nbinom.pmf(x, self.params.r, self.params.p)

    def update_params(self, data: NDArray, weights: NDArray | None = None) -> None:
        """Update negative binomial parameters using weighted method of moments.

        Uses the method of moments estimator with sample mean and variance
        to estimate r and p parameters.

        Args:
            data: Array of count data.
            weights: Array of weights for each observation. If None, all
                observations are weighted equally.

        Examples:
            >>> import numpy as np
            >>> dist = NegativeBinomialDistribution()
            >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> old_r, old_p = dist.r, dist.p
            >>> dist.update_params(data)
            >>> # Parameters should be updated
            >>> (dist.r, dist.p) != (old_r, old_p)
            True
        """
        if weights is None:
            weights = np.ones_like(data, dtype=float)

        weighted_sum = np.sum(weights)

        if weighted_sum > 0:
            weighted_mean = np.sum(weights * data) / weighted_sum
            weighted_var = np.sum(weights * (data - weighted_mean) ** 2) / weighted_sum

            if weighted_var > weighted_mean and weighted_mean > 0:
                self.p = weighted_mean / weighted_var
                self.r = weighted_mean * self.p / (1 - self.p)

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate samples from negative binomial distribution.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator for reproducible sampling.
            seed: Random seed (used only if rng is None).

        Returns:
            Array of negative binomial samples (number of failures).

        Examples:
            >>> import numpy as np
            >>> dist = NegativeBinomialDistribution(r=5.0, p=0.3)
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
        return rng.negative_binomial(self.params.r, self.params.p, n_samples)
