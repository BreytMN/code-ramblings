from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class Blob:
    """A blob configuration for generating clustered data points.

    Args:
        n_samples: Number of samples to generate for this blob
        center: Center coordinates of the blob
        cluster_std: Standard deviation of the blob (controls spread)

    Examples:
        >>> blob = Blob(n_samples=100, center=(0, 0), cluster_std=1.0)
        >>> blob.n_samples
        100
        >>> blob.center
        (0, 0)
        >>> blob.cluster_std
        1.0
    """

    n_samples: int
    center: Sequence[float]
    cluster_std: float

    def generate_samples(
        self, rng: np.random.RandomState, cluster_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate samples for this blob.

        Args:
            rng: Random number generator for reproducible results
            cluster_id: Identifier for this cluster (used in labels)

        Returns:
            Tuple of (samples, labels) where samples is shape (n_samples, n_features)
            and labels is shape (n_samples,) filled with cluster_id

        Examples:
            >>> rng = np.random.RandomState(42)
            >>> blob = Blob(n_samples=5, center=(1, 2), cluster_std=0.1)
            >>> samples, labels = blob.generate_samples(rng, cluster_id=0)
            >>> samples.shape
            (5, 2)
            >>> labels.shape
            (5,)
            >>> all(labels == 0)
            True
            >>> np.allclose(samples.mean(axis=0), [1, 2], atol=0.5)
            True
        """
        center = np.array(self.center)
        n_features = len(center)

        # Create covariance matrix (isotropic, same std in all directions)
        cov_matrix = np.eye(n_features) * (self.cluster_std**2)

        samples = rng.multivariate_normal(
            mean=center, cov=cov_matrix, size=self.n_samples
        )

        labels = np.full(self.n_samples, cluster_id)

        return samples, labels


@dataclass(frozen=True, slots=True)
class BlobDataset:
    """A dataset generator for creating blob-like clusters using numpy.

    Args:
        blobs: Sequence of Blob configurations

    Examples:
        >>> blobs = [
        ...     Blob(n_samples=50, center=(0, 0), cluster_std=1.0),
        ...     Blob(n_samples=30, center=(5, 5), cluster_std=0.5)
        ... ]
        >>> dataset = BlobDataset(blobs)
        >>> df = dataset(seed=42)
        >>> df.shape
        (80, 3)
        >>> list(df.columns)
        ['x_0', 'x_1', 'y']
        >>> df['y'].value_counts().sort_index().tolist()
        [50, 30]
    """

    blobs: Sequence[Blob]

    def __post_init__(self) -> None:
        """Validate that all blob centers have the same dimensionality."""
        if not self.blobs:
            raise ValueError("At least one blob must be provided")

        center_components = len(self.blobs[0].center)
        for i, blob in enumerate(self.blobs):
            if len(blob.center) != center_components:
                raise ValueError(
                    f"All centers must have the same number of components. "
                    f"Blob 0 has {center_components} components, "
                    f"but blob {i} has {len(blob.center)} components"
                )

    def __call__(
        self,
        seed: int | None = None,
        output_type: Literal["pandas", "numpy"] = "pandas",
    ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
        """Generate the blob dataset.

        Args:
            seed: Random seed for reproducibility
            output_type: Format of the output ("pandas" for DataFrame, "numpy" for arrays)

        Returns:
            DataFrame with feature columns (x_0, x_1, ...) and target column (y) if output_type="pandas",
            or tuple of (features, labels) arrays if output_type="numpy"

        Examples:
            >>> blobs = [Blob(5, (0, 0), 1.0), Blob(3, (2, 2), 0.5)]
            >>> dataset = BlobDataset(blobs)
            >>> df = dataset(seed=42)  # Default to pandas
            >>> df.shape
            (8, 3)
            >>> X, y = dataset(seed=42, output_type="numpy")
            >>> X.shape
            (8, 2)
            >>> y.shape
            (8,)
        """
        if output_type == "pandas":
            return self.df(seed)
        elif output_type == "numpy":
            return self.arrays(seed)
        else:
            raise ValueError(
                f"Invalid output_type: {output_type}. Must be 'pandas' or 'numpy'."
            )

    def _arrays(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate the blob dataset as numpy arrays (internal implementation).

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        rng = np.random.RandomState(seed)

        all_samples = []
        all_labels = []

        for cluster_id, blob in enumerate(self.blobs):
            samples, labels = blob.generate_samples(rng, cluster_id)
            all_samples.append(samples)
            all_labels.append(labels)

        # Combine all samples and labels
        X = np.vstack(all_samples)
        y = np.concatenate(all_labels)

        # Shuffle the data to mix clusters
        indices = rng.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def df(self, seed: int | None = None) -> pd.DataFrame:
        """Generate the blob dataset as a DataFrame.

        Args:
            seed: Random seed for reproducibility

        Returns:
            DataFrame with feature columns (x_0, x_1, ...) and target column (y)

        Examples:
            >>> blobs = [Blob(10, (0, 0), 1.0), Blob(5, (3, 3), 0.5)]
            >>> dataset = BlobDataset(blobs)
            >>> df = dataset.df(seed=42)
            >>> df.shape
            (15, 3)
            >>> (df['y'] == 0).sum().item()
            10
            >>> (df['y'] == 1).sum().item()
            5
        """
        X, y = self._arrays(seed)
        n_features = len(self.blobs[0].center)
        features = [f"x_{i}" for i in range(n_features)]

        return pd.concat(
            (
                pd.DataFrame(X, columns=features),
                pd.Series(y, name="y"),
            ),
            axis=1,
        )

    def arrays(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate the blob dataset as numpy arrays.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (features, labels) as numpy arrays

        Examples:
            >>> blobs = [Blob(5, (0, 0), 1.0), Blob(3, (2, 2), 0.5)]
            >>> dataset = BlobDataset(blobs)
            >>> X, y = dataset.arrays(seed=42)
            >>> X.shape
            (8, 2)
            >>> y.shape
            (8,)
            >>> len(np.unique(y))
            2
        """
        return self._arrays(seed)
