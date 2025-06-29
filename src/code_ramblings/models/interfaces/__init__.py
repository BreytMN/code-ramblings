from typing import Protocol, Self  # pragma: no cover

from numpy.typing import ArrayLike, NDArray  # pragma: no cover


class SklearnClassifier(Protocol):  # pragma: no cover
    """Protocol for sklearn-compatible classifiers."""

    classes_: NDArray  # shape: (n_classes,)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the classifier to training data.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
        """
        ...

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels for samples.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        ...

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities for samples.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples, n_classes)
        """
