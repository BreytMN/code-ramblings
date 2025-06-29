import warnings
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import pandas as pd  # type: ignore[import-untyped]
    from pandas import DataFrame, Series  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    # fmt: off
    class DataFrame: ...  # type: ignore[no-redef]
    class Series: ...  # type: ignore[no-redef]
    pd = None


try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    # fmt: off
    class Tensor: ...  # type: ignore[no-redef]
    class torch_dtype: ...  # type: ignore[no-redef]
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class DataContainer:
    """Data container for ML training and inference pipelines.

    A frozen dataclass that encapsulates feature and target data for machine learning
    workflows, providing convenient methods to convert between pandas, numpy, and torch
    formats. Ensures data consistency and provides a unified interface for different
    ML frameworks.

    Attributes:
        X: Training features as 2D numpy array, shape (n_samples, n_features)
        y: Training labels as 1D numpy array, shape (n_samples,)

    Examples:
        >>> import numpy as np
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y = np.array([0, 1, 0])
        >>> container = DataContainer(X=X, y=y)
        >>> container.X.shape
        (3, 2)
        >>> container.y.shape
        (3,)

        >>> # Access column names (auto-generated if not provided)
        >>> container._X_cols
        ['x_0', 'x_1']
        >>> container._y_col
        'y'
    """

    X: NDArray
    y: NDArray

    _X_cols: list[str] = field(init=False)
    _y_col: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate input arrays and initialize column names."""
        if not isinstance(self.X, np.ndarray) or not isinstance(self.y, np.ndarray):
            raise TypeError(
                "X and y must be numpy arrays, "
                f"got X as {type(self.X)} and y as {type(self.y)}."
            )

        if (n := len(self.X.shape)) != 2:
            raise ValueError(f"X must be a 2d array, got shape of X as {n}.")

        if (n := len(self.y.shape)) != 1:
            raise ValueError(f"y must be a 1d array, got shape of y as {n}.")

        if (n := self.X.shape[0]) != (m := self.y.shape[0]):
            raise ValueError(
                "Both arrays must have same length, "
                f"X has length {n} and y has length {m}"
            )

        if not hasattr(self, "_X_cols"):
            X_cols = [f"x_{i}" for i in range(self.X.shape[1])]
            object.__setattr__(self, "_X_cols", X_cols)
        if not hasattr(self, "_y_col"):
            object.__setattr__(self, "_y_col", "y")

    def __repr__(self) -> str:
        """Return string representation showing data dimensions and column info.

        Returns:
            A formatted string showing the shape of X and y arrays, along with
            column names (truncated if more than 5 features).

        Examples:
            >>> import numpy as np
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> container = DataContainer(X=X, y=y)
            >>> print(repr(container))  # doctest: +ELLIPSIS
            DataContainer(
              X: shape=(2, 2), columns=['x_0', 'x_1']
              y: shape=(2,), name='y'
            )
        """
        n_samples, n_features = self.X.shape

        # Truncate column names if too many
        if len(self._X_cols) <= 5:
            X_cols_str = str(self._X_cols)
        else:
            first_cols = [repr(c) for c in self._X_cols[:3]]
            last_col = repr(self._X_cols[-1])
            X_cols_str = f"[{', '.join(first_cols)}, ..., {last_col}]"

        return (
            f"DataContainer(\n"
            f"  X: shape=({n_samples}, {n_features}), columns={X_cols_str}\n"
            f"  y: shape=({n_samples},), name={repr(self._y_col)}\n"
            f")"
        )

    @classmethod
    def from_array(
        cls,
        X: ArrayLike,
        y: ArrayLike,
        X_names: list[str] | None = None,
        y_name: str | None = None,
    ) -> Self:
        """Create DataContainer from array-like objects with optional column names.

        Args:
            X: Feature array-like object, shape (n_samples, n_features)
            y: Target array-like object, shape (n_samples,)
            X_names: Optional list of feature column names. If None, auto-generates
                names like ['x_0', 'x_1', ...]
            y_name: Optional target column name. If None, defaults to 'y'

        Returns:
            DataContainer instance with the provided data and names.

        Raises:
            RuntimeError: If X_names length doesn't match number of features in X.

        Examples:
            >>> import numpy as np
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> container = DataContainer.from_array(X, y)
            >>> container._X_cols
            ['x_0', 'x_1']

            >>> # With custom names
            >>> container = DataContainer.from_array(
            ...     X, y, X_names=['feature1', 'feature2'], y_name='target'
            ... )
            >>> container._X_cols
            ['feature1', 'feature2']
            >>> container._y_col
            'target'
        """
        X, y = np.asarray(X), np.asarray(y)
        instance = cls(X=X, y=y)

        if X_names is not None:
            if (n := len(X_names)) != (m := X.shape[1]):
                raise RuntimeError(
                    "List X_names must have the same length as size of second "
                    f"dimension of array X, got {n} names for {m} columns."
                )
            object.__setattr__(instance, "_X_cols", X_names)

        if y_name is not None:
            object.__setattr__(instance, "_y_col", y_name)

        return instance

    @classmethod
    def from_pandas(
        cls,
        df: ArrayLike,
        X_cols: list[str] | None = None,
        y_col: str | None = None,
    ) -> Self:
        """Create DataContainer from pandas DataFrame.

        Args:
            df: Pandas DataFrame containing features and target data
            X_cols: List of column names to use as features. If None and y_col
                is provided, uses all columns except y_col
            y_col: Column name to use as target. If None and X_cols is provided,
                uses the first remaining column. If both None, uses last column
                as target and all others as features

        Returns:
            DataContainer instance with data from the DataFrame.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ImportError: If pandas is not installed.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> df = pd.DataFrame({
            ...     'feature1': [1, 2, 3],
            ...     'feature2': [4, 5, 6],
            ...     'target': [0, 1, 0]
            ... })
            >>> container = DataContainer.from_pandas(df, y_col='target')
            >>> container._X_cols
            ['feature1', 'feature2']
            >>> container._y_col
            'target'

            >>> # Auto-detect (last column as target)
            >>> container = DataContainer.from_pandas(df)  # doctest: +SKIP
            >>> container._y_col  # doctest: +SKIP
            'target'
        """
        try:
            import pandas as pd

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"df must be a pandas dataframe, got {type(df)}")
        except ImportError:  # pragma: no cover
            raise ImportError("Missing library: pandas")

        if X_cols is None and y_col is None:
            warnings.warn(
                "Features and target will be assumed by column position. "
                "Column at the last position will be set as target, "
                "everything else will be set as features.",
                UserWarning,
            )

            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            X_cols, y_col = X.columns.tolist(), y.name
        else:
            if X_cols is None and y_col is not None:
                X_cols = [c for c in df.columns if c != y_col]
            if y_col is None and X_cols is not None:
                y_col = [c for c in df.columns if c not in X_cols][0]
            X, y = df.filter(X_cols), df.get(y_col)

        instance = cls(X=X.to_numpy(), y=y.to_numpy())
        object.__setattr__(instance, "_X_cols", X_cols)
        object.__setattr__(instance, "_y_col", y_col)

        return instance

    def as_array(self) -> tuple[NDArray, NDArray]:
        """Return data as numpy arrays.

        Returns:
            Tuple of (X, y) as numpy arrays.

        Examples:
            >>> import numpy as np
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> container = DataContainer(X=X, y=y)
            >>> X_np, y_np = container.as_array()
            >>> X_np.shape
            (2, 2)
            >>> y_np.shape
            (2,)
        """
        return np.asarray(self.X), np.asarray(self.y)

    def as_pandas(self) -> tuple[DataFrame, Series]:
        """Return data as pandas DataFrame and Series.

        Returns:
            Tuple of (DataFrame, Series) with feature and target data.
            Uses stored column names for proper labeling.

        Raises:
            ImportError: If pandas is not installed.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> X = np.array([[1, 2], [3, 4]])
            >>> y = np.array([0, 1])
            >>> container = DataContainer.from_array(
            ...     X, y, X_names=['feat1', 'feat2'], y_name='target'
            ... )
            >>> df_X, ser_y = container.as_pandas()
            >>> list(df_X.columns)
            ['feat1', 'feat2']
            >>> ser_y.name
            'target'
        """
        if pd is None:  # pragma: no cover
            raise ImportError("Missing library: pandas")

        return (
            pd.DataFrame(self.X, columns=self._X_cols),
            pd.Series(self.y, name=self._y_col),
        )

    def as_tensor(self) -> tuple[Tensor, Tensor]:
        """Return data as PyTorch tensors.

        Returns:
            Tuple of (Tensor, Tensor) with feature and target data.

        Raises:
            ImportError: If PyTorch is not installed.

        Examples:
            >>> import numpy as np
            >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> y = np.array([0, 1])
            >>> container = DataContainer(X=X, y=y)
            >>> X_torch, y_torch = container.as_tensor()
            >>> X_torch.shape
            torch.Size([2, 2])
            >>> y_torch.shape
            torch.Size([2])
        """
        if torch is None:  # pragma: no cover
            raise ImportError("Missing library: pytorch")

        return torch.tensor(self.X), torch.tensor(self.y)
