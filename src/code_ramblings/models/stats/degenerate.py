from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._generics import CountDistribution, CountParams


@dataclass
class DegenerateParams(CountParams):
    """Parameters for degenerate distribution.

    Attributes:
        point: The point where all probability mass is concentrated.
    """

    point: float


class DegenerateDistribution(CountDistribution[DegenerateParams]):
    """Degenerate distribution that places all probability mass at a single point.

    This distribution is also known as a point mass distribution. All probability
    is concentrated at a single value, making it useful for modeling scenarios
    where a specific outcome is certain to occur.

    Examples:
        >>> import numpy as np
        >>> dist = DegenerateDistribution(point=0.0)
        >>> dist.point
        0.0
    """

    def __init__(self, point: float = 0.0) -> None:
        """Initialize degenerate distribution.

        Args:
            point: The point where all probability mass is concentrated.

        Examples:
            >>> dist = DegenerateDistribution()
            >>> dist.point
            0.0
            >>> dist2 = DegenerateDistribution(point=5.0)
            >>> dist2.point
            5.0
        """
        self.params = DegenerateParams(point=point)

    def __repr__(self) -> str:
        return f"DegenerateDistribution(point={self.params.point})"

    @property
    def point(self) -> float:
        """The point where all probability mass is concentrated."""
        return self.params.point

    @point.setter
    def point(self, v: float) -> None:
        self.params.point = float(v)

    def pmf(self, x: NDArray) -> NDArray:
        """Compute degenerate distribution probability mass function.

        Args:
            x: Array of values.

        Returns:
            Array of probability mass values (1.0 where x equals point, 0.0 elsewhere).

        Examples:
            >>> import numpy as np
            >>> dist = DegenerateDistribution(point=0.0)
            >>> x = np.array([0, 1, 2, 0])
            >>> probs = dist.pmf(x)
            >>> np.array_equal(probs, [1.0, 0.0, 0.0, 1.0])
            True
        """
        return (x == self.params.point).astype(float)

    def update_params(self, data: NDArray, weights: NDArray | None = None) -> None:
        """Update degenerate distribution parameter using weighted mode.

        Finds the value with the highest weighted frequency and sets it as
        the new point mass location.

        Args:
            data: Array of count data.
            weights: Array of weights for each observation. If None, all
                observations are weighted equally.

        Examples:
            >>> import numpy as np
            >>> dist = DegenerateDistribution()
            >>> data = np.array([0, 1, 0, 2])
            >>> weights = np.array([0.8, 0.1, 0.9, 0.1])  # 0 has highest total weight
            >>> dist.update_params(data, weights)
            >>> dist.point
            0.0
        """
        if weights is None:
            weights = np.ones_like(data, dtype=float)

        unique_vals = np.unique(data)
        weighted_counts = np.array(
            [np.sum(weights[data == val]) for val in unique_vals]
        )

        self.point = float(unique_vals[np.argmax(weighted_counts)])

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate samples from degenerate distribution.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator (unused but required by protocol).
            seed: Random seed (unused but required by protocol).

        Returns:
            Array of samples (all equal to the point mass location).

        Examples:
            >>> import numpy as np
            >>> dist = DegenerateDistribution(point=0.0)
            >>> samples = dist.sample(5, seed=42)
            >>> np.array_equal(samples, [0, 0, 0, 0, 0])
            True
            >>> samples.dtype in [np.int32, np.int64]
            True
        """
        return np.full(n_samples, self.params.point, dtype=int)
