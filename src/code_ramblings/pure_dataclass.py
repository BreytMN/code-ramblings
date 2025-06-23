from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import pandas as pd  # type: ignore
from numpy.typing import NDArray

try:
    import torch  # type: ignore
    from torch import Tensor
    from torch import dtype as torch_dtype

except ImportError:  # pragma: no cover

    class Tensor: ...  # type: ignore [no-redef]

    class torch_dtype: ...  # type: ignore [no-redef]

    torch = None  # type: ignore [assignment]


SetType = Literal["train", "calib", "valid", "test"]


@dataclass(frozen=True, slots=True)
class DataContainer:
    """Data container for ML training and inference pipelines.

    A frozen dataclass that encapsulates feature and target data for machine learning
    workflows, providing convenient methods to convert between pandas, numpy, and torch
    formats while maintaining data integrity and type safety.

    Attributes:
        x: Feature data as a pandas DataFrame.
        y: Target data as a pandas Series.
        _set_type: Internal attribute to store the type of dataset (e.g., "train", "test").

    Examples:
        Basic usage with pandas DataFrame:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import torch
        >>> df = pd.DataFrame({
        ...     "feature_1": [1.0, 2.0, 3.0],
        ...     "feature_2": [0.5, 1.5, 2.5],
        ...     "target": [10.0, 20.0, 30.0]
        ... })
        >>> container = DataContainer.from_pandas_df(
        ...     df, ["feature_1", "feature_2"], "target", set_type="train"
        ... )
        >>> container.set_type
        'train'

        Converting to different formats:

        >>> X_pd, y_pd = container.as_pandas_objects()
        >>> X_pd.shape
        (3, 2)
        >>> y_pd.name
        'target'

        >>> X_np, y_np = container.as_numpy_array(np.float32)
        >>> X_np.dtype
        dtype('float32')
        >>> X_np.shape
        (3, 2)

        >>> X_torch, y_torch = container.as_torch_tensor(torch.float64)
        >>> X_torch.dtype
        torch.float64
        >>> X_torch.shape
        torch.Size([3, 2])
    """

    x: pd.DataFrame
    y: pd.Series

    _set_type: SetType

    @property
    def set_type(self) -> SetType:
        """Indicates the type of dataset held by the container.

        Returns:
            The type of dataset, e.g., "train", "calib", "valid", or "test".

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"x": [1.0], "y": [2.0]})
            >>> train_container = DataContainer.from_pandas_df(df, ["x"], "y", set_type="train")
            >>> train_container.set_type
            'train'
            >>> test_container = DataContainer.from_pandas_df(df, ["x"], "y", set_type="test")
            >>> test_container.set_type
            'test'
        """

        return self._set_type

    @classmethod
    def from_pandas_df(
        cls,
        df: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
        set_type: SetType,
    ) -> Self:
        """Create a DataContainer from a pandas DataFrame.

        Args:
            df: Source DataFrame containing both features and target.
            x_cols: List of column names to use as features.
            y_col: Column name to use as target variable.
            set_type: The type of dataset this container represents (e.g., "train", "test").

        Returns:
            New DataContainer instance with the specified features and target.

        Raises:
            KeyError: If any specified columns are missing from the DataFrame.

        Examples:
            Creating training data:

            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     "age": [25, 30, 35],
            ...     "income": [50000, 60000, 70000],
            ...     "approved": [0, 1, 1]
            ... })
            >>> container = DataContainer.from_pandas_df(
            ...     df, ["age", "income"], "approved", set_type="train"
            ... )
            >>> container.x.columns.tolist()
            ['age', 'income']
            >>> container.y.name
            'approved'
            >>> container.set_type
            'train'

            Creating test data:

            >>> test_container = DataContainer.from_pandas_df(
            ...     df, ["age", "income"], "approved", set_type="test"
            ... )
            >>> test_container.set_type
            'test'

            Error raising for missing cols

            >>> container = DataContainer.from_pandas_df(
            ...     df, ["age", "income", "location"], "approved", set_type="train"
            ... )
            Traceback (most recent call last):
            KeyError: 'columns missing from dataframe: location'
            >>> container = DataContainer.from_pandas_df(
            ...     df, ["age", "income"], "status", set_type="train"
            ... )
            Traceback (most recent call last):
            KeyError: 'columns missing from dataframe: status'
        """
        missing_x_cols = [c for c in x_cols if c not in df.columns]
        if missing_x_cols:
            msg = f"columns missing from dataframe: {' ,'.join(missing_x_cols)}"
            raise KeyError(msg)

        missing_y_col = y_col not in df.columns
        if missing_y_col:
            msg = f"columns missing from dataframe: {y_col}"
            raise KeyError(msg)

        return cls(
            x=df.filter(x_cols),
            y=df.get(y_col),
            _set_type=set_type,
        )

    def as_pandas_objects(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return the feature and target data as pandas objects.

        Returns:
            Tuple of (features DataFrame, target Series).

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"x1": [1, 2], "x2": [3, 4], "y": [5, 6]})
            >>> container = DataContainer.from_pandas_df(df, ["x1", "x2"], "y", set_type="train")
            >>> X, y = container.as_pandas_objects()
            >>> isinstance(X, pd.DataFrame)
            True
            >>> isinstance(y, pd.Series)
            True
            >>> X.shape
            (2, 2)
            >>> len(y)
            2
        """
        return self.x, self.y

    def as_numpy_array(
        self, dtype: type[np.floating] | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Convert feature and target data to numpy arrays.

        Args:
            dtype: Numpy data type for the arrays. If None, uses pandas default conversion.

        Returns:
            Tuple of (features array, target array) as numpy NDArrays.

        Examples:
            Default conversion:

            >>> import numpy as np
            >>> import pandas as pd
            >>> df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
            >>> container = DataContainer.from_pandas_df(df, ["x"], "y", set_type="train")
            >>> X, y = container.as_numpy_array()
            >>> X.shape
            (2, 1)
            >>> y.shape
            (2,)

            With specific dtype:

            >>> X_f32, y_f32 = container.as_numpy_array(np.float32)
            >>> X_f32.dtype
            dtype('float32')
            >>> y_f32.dtype
            dtype('float32')
        """
        return (
            self.x.to_numpy(dtype=dtype),
            self.y.to_numpy(dtype=dtype),
        )

    def as_torch_tensor(
        self, dtype: torch_dtype | None = None
    ) -> tuple[Tensor, Tensor]:
        """Convert feature and target data to PyTorch tensors.

        Args:
            dtype: PyTorch data type for the tensors. If None, uses default conversion.

        Returns:
            Tuple of (features tensor, target tensor) as PyTorch Tensors.

        Examples:
            Default conversion:

            >>> import pandas as pd
            >>> import torch
            >>> df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [5.0, 6.0]})
            >>> container = DataContainer.from_pandas_df(df, ["x1", "x2"], "y", set_type="train")
            >>> X_tensor, y_tensor = container.as_torch_tensor()
            >>> X_tensor.shape
            torch.Size([2, 2])
            >>> y_tensor.shape
            torch.Size([2])

            With specific dtype:

            >>> X_f64, y_f64 = container.as_torch_tensor(torch.float64)
            >>> X_f64.dtype
            torch.float64
            >>> y_f64.dtype
            torch.float64
        """
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is not installed.")

        return (
            torch.tensor(self.x.to_numpy(), dtype=dtype),
            torch.tensor(self.y.to_numpy(), dtype=dtype),
        )
