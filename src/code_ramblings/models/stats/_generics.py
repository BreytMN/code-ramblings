from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray


@dataclass
class CountParams:
    """Base class for count distribution parameters."""

    ...


CountParams_ = TypeVar("CountParams_", bound=CountParams)


class CountDistribution(Protocol[CountParams_]):  # pragma: no cover
    """Protocol for discrete count distributions.

    This protocol defines the interface that count distributions must implement
    to support parameter estimation, probability computation, and sampling.
    Distributions implementing this protocol can be used in various statistical
    modeling contexts.
    """

    params: CountParams_

    def pmf(self, x: NDArray) -> NDArray:
        """Compute probability mass function.

        Args:
            x: Array of non-negative integer count data.

        Returns:
            Array of probability mass values corresponding to input data.

        Examples:
            >>> import numpy as np
            >>> # This is a protocol, so we can't instantiate it directly
            >>> # but implementations would work like:
            >>> # dist.pmf(np.array([0, 1, 2, 3]))
        """
        ...

    def update_params(self, data: NDArray, weights: NDArray | None = None) -> None:
        """Update parameters using weighted maximum likelihood estimation.

        Args:
            data: Array of count data.
            weights: Array of weights for each observation. If None, all
                observations are weighted equally.

        Examples:
            >>> import numpy as np
            >>> # This is a protocol, so we can't instantiate it directly
            >>> # but implementations would work like:
            >>> # dist.update_params(np.array([1, 2, 3]), np.array([0.5, 0.8, 0.3]))
        """
        ...

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> NDArray:
        """Generate random samples from the distribution.

        Args:
            n_samples: Number of samples to generate.
            rng: Random number generator for reproducible sampling.
            seed: Random seed (used only if rng is None).

        Returns:
            Array of randomly generated samples from the distribution.

        Examples:
            >>> import numpy as np
            >>> # This is a protocol, so we can't instantiate it directly
            >>> # but implementations would work like:
            >>> # samples = dist.sample(100, seed=42)
        """
        ...
