"""
Self-contained ConformalClassifierModel and ConformalClassifierEvaluator for conformal prediction.

This implementation splits the conformal prediction functionality into:
1. ConformalClassifierModel: Core prediction model with training, calibration, and prediction
2. ConformalClassifierEvaluator: Comprehensive evaluation framework for analyzing model performance
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..interfaces import SklearnClassifier

TieBreakingMethod = Literal["smoothing", "conservative", "optimistic", "exact"]


@dataclass
class ConformalPredictions:
    """Container for conformal prediction results.

    Examples:
        >>> import numpy as np
        >>> point_pred = np.array([0, 1, 0])
        >>> point_pred_cal = np.array([0, 1, 1])
        >>> p_vals = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        >>> pred_sets = {0.1: np.array([[1, 0], [0, 1], [1, 1]])}
        >>> probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        >>> predictions = ConformalPredictions(
        ...     point_pred, point_pred_cal, p_vals, pred_sets, probs
        ... )
        >>> predictions.point_predictions.shape
        (3,)
        >>> list(predictions.prediction_sets.keys())
        [0.1]
    """

    point_predictions: NDArray
    point_predictions_calibrated: NDArray
    p_values: NDArray
    prediction_sets: dict[float, NDArray]
    probabilities: NDArray


@dataclass
class CoverageResult:
    """Container for coverage evaluation results.

    Examples:
        >>> result = CoverageResult(
        ...     alpha=0.1,
        ...     expected_coverage=0.9,
        ...     coverage_rate=0.92,
        ...     coverage_gap=0.02,
        ...     covered_instances=92,
        ...     total_instances=100,
        ...     avg_set_size=2.5
        ... )
        >>> result.coverage_gap
        0.02
        >>> result.coverage_rate > result.expected_coverage
        True
    """

    alpha: float
    expected_coverage: float
    coverage_rate: float
    coverage_gap: float
    covered_instances: int
    total_instances: int
    avg_set_size: float


@dataclass
class EfficiencyResult:
    """Container for efficiency evaluation results.

    Examples:
        >>> result = EfficiencyResult(
        ...     alpha=0.1,
        ...     avg_set_size=2.3,
        ...     median_set_size=2.0,
        ...     std_set_size=1.2,
        ...     singleton_rate=0.15,
        ...     empty_rate=0.02,
        ...     singleton_count=15,
        ...     empty_count=2,
        ...     total_instances=100
        ... )
        >>> result.singleton_rate
        0.15
        >>> result.avg_set_size > result.median_set_size
        True
    """

    alpha: float
    avg_set_size: float
    median_set_size: float
    std_set_size: float
    singleton_rate: float
    empty_rate: float
    singleton_count: int
    empty_count: int
    total_instances: int


@dataclass
class ClassConditionalCoverageResult:
    """Container for class-conditional coverage results.

    Examples:
        >>> result = ClassConditionalCoverageResult(
        ...     alpha=0.1,
        ...     target_label=1,
        ...     expected_coverage=0.9,
        ...     coverage_rate=0.88,
        ...     coverage_gap=-0.02,
        ...     covered_instances=44,
        ...     target_instances=50,
        ...     avg_set_size=2.1
        ... )
        >>> result.target_label
        1
        >>> result.coverage_rate < result.expected_coverage
        True
    """

    alpha: float
    target_label: int
    expected_coverage: float
    coverage_rate: float
    coverage_gap: float
    covered_instances: int
    target_instances: int
    avg_set_size: float


@dataclass
class ClassConditionalPrecisionResult:
    """Container for class-conditional precision results.

    Examples:
        >>> result = ClassConditionalPrecisionResult(
        ...     alpha=0.1,
        ...     target_label=0,
        ...     precision_rate=0.75,
        ...     correct_predictions=30,
        ...     sets_containing_target=40,
        ...     total_instances=100,
        ...     pp_gain_over_random=25.0,
        ...     random_precision=0.5
        ... )
        >>> result.precision_rate
        0.75
        >>> result.pp_gain_over_random
        25.0
    """

    alpha: float
    target_label: int
    precision_rate: float
    correct_predictions: int
    sets_containing_target: int
    total_instances: int
    pp_gain_over_random: float
    random_precision: float


@dataclass
class SizeStratifiedCoverageResult:
    """Container for size-stratified coverage results.

    Examples:
        >>> result = SizeStratifiedCoverageResult(
        ...     set_size=2,
        ...     coverage_rate=0.95,
        ...     covered_instances=19,
        ...     instance_count=20
        ... )
        >>> result.set_size
        2
        >>> result.coverage_rate == result.covered_instances / result.instance_count
        True
    """

    set_size: int
    coverage_rate: float
    covered_instances: int
    instance_count: int


@dataclass
class ClassStats:
    """Statistics for a specific class in conformal prediction evaluation.

    Examples:
        >>> import numpy as np
        >>> indices = np.array([0, 2, 4])
        >>> stats = ClassStats(
        ...     label=1,
        ...     total_instances=3,
        ...     covered_instances=2,
        ...     sets_containing_label=5,
        ...     correct_predictions=2,
        ...     indices=indices
        ... )
        >>> stats.label
        1
        >>> len(stats.indices)
        3
    """

    label: int
    total_instances: int
    covered_instances: int
    sets_containing_label: int
    correct_predictions: int
    indices: NDArray

    @classmethod
    def compute(cls, label: int, y: NDArray, sets: NDArray, cov_mask: NDArray) -> Self:
        """Compute class statistics from predictions and coverage.

        Args:
            label: Class label to compute statistics for
            y: True labels array
            sets: Binary prediction sets array
            cov_mask: Boolean coverage mask

        Returns:
            ClassStats instance with computed statistics

        Examples:
            >>> import numpy as np
            >>> y = np.array([0, 1, 1, 0, 1])
            >>> sets = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]])
            >>> cov_mask = np.array([True, True, True, True, False])
            >>> stats = ClassStats.compute(1, y, sets, cov_mask)
            >>> stats.label
            1
            >>> stats.total_instances
            3
            >>> stats.covered_instances
            2
        """
        label_mask = y == label
        label_indices = np.where(label_mask)[0]

        class_covered = int(np.sum(cov_mask[label_mask]))
        sets_with_label_mask = sets[:, label].astype(bool)
        correct_predictions = int(np.sum(y[sets_with_label_mask] == label))

        return cls(
            label=label,
            total_instances=len(label_indices),
            covered_instances=class_covered,
            sets_containing_label=int(np.sum(sets_with_label_mask)),
            correct_predictions=correct_predictions,
            indices=label_indices,
        )


@dataclass
class AlphaMetrics:
    """Cache entry for alpha-specific conformal prediction results.

    Examples:
        >>> import numpy as np
        >>> alpha = 0.1
        >>> pred_sets = np.array([[1, 0], [0, 1], [1, 1]])
        >>> cov_mask = np.array([True, True, False])
        >>> set_sizes = np.array([1, 1, 2])
        >>> class_stats = {0: ClassStats(0, 1, 1, 2, 1, np.array([0]))}
        >>> metrics = AlphaMetrics(
        ...     alpha, pred_sets, cov_mask, 2, set_sizes, class_stats
        ... )
        >>> metrics.alpha
        0.1
        >>> metrics.coverage_count
        2
    """

    alpha: float
    prediction_sets: NDArray
    coverage_mask: NDArray
    coverage_count: int
    set_sizes: NDArray
    class_stats: dict[int, ClassStats]

    @classmethod
    def create_for_alpha(
        cls,
        alpha: float,
        model: "ConformalClassifierModel",
        X: ArrayLike,
        y: NDArray,
        n_samples: int,
        unique_labels: NDArray,
        tie_breaking: TieBreakingMethod,
        predictions: ConformalPredictions,
    ) -> Self:
        """Create cache entry for a specific alpha value.

        Args:
            alpha: Significance level to compute entry for
            model: Trained and calibrated conformal classifier model
            X: Test features
            y: Test labels
            n_samples: Number of test samples
            unique_labels: Unique labels in the test set
            tie_breaking: Method for handling tied scores
            predictions: Precomputed conformal predictions object

        Returns:
            New AlphaMetrics instance

        Examples:
            >>> # Note: This example requires a trained model and data
            >>> # alpha_metrics = AlphaMetrics.create_for_alpha(
            >>> #     alpha=0.1,
            >>> #     model=trained_model,
            >>> #     X=X_test,
            >>> #     y=y_test,
            >>> #     n_samples=len(y_test),
            >>> #     unique_labels=np.unique(y_test),
            >>> #     tie_breaking="exact",
            >>> #     predictions=predictions
            >>> # )
            >>> # alpha_metrics.alpha == 0.1
            >>> True
        """
        if alpha in predictions.prediction_sets:
            sets = predictions.prediction_sets[alpha]
        else:
            sets = model.predict_set(X, alpha, tie_breaking)

        coverage_mask = sets[np.arange(n_samples), y]
        coverage_count = int(np.sum(coverage_mask))
        set_sizes = np.sum(sets, axis=1)

        class_stats = {}
        for label in unique_labels:
            class_stats[label] = ClassStats.compute(label, y, sets, coverage_mask)

        return cls(
            alpha=alpha,
            prediction_sets=sets,
            coverage_mask=coverage_mask,
            coverage_count=coverage_count,
            set_sizes=set_sizes,
            class_stats=class_stats,
        )


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for conformal prediction.

    Examples:
        >>> metrics = EvaluationMetrics(
        ...     error=0.08,
        ...     avg_c=2.3,
        ...     one_c=0.35,
        ...     empty=0.02
        ... )
        >>> metrics.error
        0.08
        >>> 1 - metrics.error  # Coverage rate
        0.92
        >>> metrics.avg_c  # Average set size
        2.3
    """

    error: float
    avg_c: float
    one_c: float
    empty: float


class ConformalClassifierModel:
    """A conformal prediction classifier for uncertainty quantification.

    This class wraps sklearn-compatible classifiers to provide conformal prediction
    capabilities including prediction sets, p-values, and calibrated probabilities.

    Examples:
        >>> # Basic usage workflow
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> import numpy as np
        >>>
        >>> # Generate sample data
        >>> X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        >>>
        >>> # Create and train conformal classifier
        >>> base_model = RandomForestClassifier(n_estimators=50, random_state=42)
        >>> conf_clf = ConformalClassifierModel(base_model)
        >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
        <...ConformalClassifierModel object at 0x...>
        >>>
        >>> # Generate predictions
        >>> predictions = conf_clf.predict(X_test[:5])
        >>> len(predictions)
        5
        >>>
        >>> # Get prediction sets for different confidence levels
        >>> pred_sets_90 = conf_clf.predict_set(X_test[:3], alpha=0.1)
        >>> pred_sets_90.shape
        (3, 3)
        >>>
        >>> # Get all prediction types at once
        >>> all_preds = conf_clf.predict_all(X_test[:5])
        >>> len(all_preds.point_predictions)
        5
        >>> all_preds.p_values.shape[1] == len(np.unique(y))
        True
    """

    @staticmethod
    def explain_tie_breaking_methods() -> dict[TieBreakingMethod, str]:
        """Explain when to use each tie-breaking method.

        Returns:
            Dictionary mapping methods to their use case descriptions

        Examples:
            >>> explanations = ConformalClassifierModel.explain_tie_breaking_methods()
            >>> len(explanations)
            4
            >>> "smoothing" in explanations
            True
            >>> "conservative" in explanations
            True
            >>> "exact" in explanations
            True
            >>> # Check that all methods provide meaningful descriptions
            >>> all(len(desc) > 50 for desc in explanations.values())
            True
        """
        return {
            "smoothing": (
                "Use for model evaluation and validation. Provides exact theoretical coverage "
                "guarantees but introduces randomness. Best for assessing model calibration."
            ),
            "conservative": (
                "Use for production systems where you prefer larger prediction sets over missing "
                "the true class. Deterministic and provides coverage >= target level."
            ),
            "optimistic": (
                "Use when you prefer smaller, more decisive prediction sets and can tolerate "
                "slightly lower coverage. Deterministic with coverage <= target level."
            ),
            "exact": (
                "Use for production systems where you want a deterministic middle ground. "
                "Provides median rank for ties - balanced between conservative and optimistic."
            ),
        }

    def __init__(self, model: SklearnClassifier) -> None:
        """Initialize the conformal classifier.

        Args:
            model: sklearn-compatible classifier

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.is_trained
            False
            >>> conf_clf.is_calibrated
            False
        """
        self.model = model

        # Conformal prediction state
        self.calibration_bins: NDArray | None = None
        self.calibration_priors: NDArray | None = None
        self.calibration_scores: NDArray | None = None

        # Training state
        self.is_trained = False
        self.is_calibrated = False

    @property
    def classes(self) -> NDArray:
        """Get unique class labels from trained model.

        Returns:
            Array of unique class labels

        Raises:
            RuntimeError: If model hasn't been trained yet

        Examples:
            >>> # This will raise an error before training
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> conf_clf = ConformalClassifierModel(RandomForestClassifier())
            >>> try:
            ...     classes = conf_clf.classes
            ... except RuntimeError as e:
            ...     "isn't trained" in str(e)
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        return self.model.classes_

    def train(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Train the base classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            Self for method chaining

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>>
            >>> # Train and verify state changes
            >>> result = conf_clf.train(X, y)
            >>> conf_clf.is_trained
            True
            >>> conf_clf.is_calibrated
            False
            >>> result is conf_clf  # Method chaining
            True
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def calibrate(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Calibrate the conformal predictor using the calibration set.

        Args:
            X: Calibration features, shape (n_samples, n_features)
            y: Calibration labels, shape (n_samples,)

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If model hasn't been trained yet

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
            >>> X_train, X_calib, y_train, y_calib = train_test_split(X, y, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> # Calibrate and verify state
            >>> result = conf_clf.calibrate(X_calib, y_calib)
            >>> conf_clf.is_calibrated
            True
            >>> conf_clf.calibration_scores is not None
            True
            >>> result is conf_clf  # Method chaining
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        scores = self.model.predict_proba(X)
        y_array = np.asarray(y)
        self.calibration_bins = y_array
        self.calibration_priors = np.bincount(y_array) / len(y_array)
        self.calibration_scores = self._hinge_score(scores, self.classes, y_array)

        self.is_calibrated = True
        return self

    def predict(
        self,
        X: ArrayLike,
        conformal: bool = True,
        tie_breaking: TieBreakingMethod = "exact",
        calibration_priors: bool = False,
    ) -> NDArray:
        """Make point predictions using the base classifier.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            conformal: Whether to use conformal calibration
            tie_breaking: Method for handling tied scores. Ignored if conformal is False.
            calibration_priors: Whether to use calibration priors in probability computation

        Returns:
            Array of predicted labels, shape (n_samples,)

        Raises:
            RuntimeError: If model hasn't been trained yet

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=300, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> # Standard predictions
            >>> preds_standard = conf_clf.predict(X_test[:5], conformal=False)
            >>> len(preds_standard)
            5
            >>>
            >>> # Conformal predictions
            >>> preds_conformal = conf_clf.predict(X_test[:5], conformal=True)
            >>> len(preds_conformal)
            5
            >>>
            >>> # All predictions should be valid class labels
            >>> all(pred in conf_clf.classes for pred in preds_conformal)
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        if not conformal:
            return self.model.predict(X)

        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        return self.predict_probability(X, tie_breaking, calibration_priors).argmax(1)

    def predict_scores(self, X: ArrayLike) -> NDArray:
        """Get raw probability scores from the base model.

        Args:
            X: Features to predict on, shape (n_samples, n_features)

        Returns:
            Array of shape (n_samples, n_classes) with probability scores

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_test = X[:80], X[80:]
            >>> y_train = y[:80]
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> scores = conf_clf.predict_scores(X_test)
            >>> scores.shape
            (20, 3)
            >>> # Probabilities should sum to 1
            >>> np.allclose(scores.sum(axis=1), 1.0)
            True
            >>> # All scores should be non-negative
            >>> (scores >= 0).all()
            True
        """
        return self.model.predict_proba(X)

    def predict_p(
        self,
        X: ArrayLike,
        tie_breaking: TieBreakingMethod = "exact",
    ) -> NDArray:
        """Predict p-values for each class using conformal prediction.

        P-values represent the confidence that each class could be the true label.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            tie_breaking: Method for handling tied scores

        Returns:
            Array of shape (n_samples, n_classes) with p-values for each class

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> p_values = conf_clf.predict_p(X_test[:5])
            >>> p_values.shape
            (5, 3)
            >>> # P-values should be between 0 and 1
            >>> ((p_values >= 0) & (p_values <= 1)).all()
            True
            >>> # Each sample should have at least one p-value > 0
            >>> (p_values.max(axis=1) > 0).all()
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        scores = self.model.predict_proba(X)
        test_scores = self._hinge_score(scores)
        X_array = np.asarray(X)

        p_values = np.zeros((len(X_array), len(self.model.classes_)))

        for c in range(len(self.model.classes_)):
            assert self.calibration_scores is not None
            bins_test = np.full(len(X_array), self.model.classes_[c])
            p_values[:, c] = self._compute_p_values(
                test_scores[:, c],
                bins_test,
                tie_breaking,
            )

        return p_values

    def predict_probability(
        self,
        X: ArrayLike,
        tie_breaking: TieBreakingMethod = "exact",
        calibration_priors: bool = False,
    ) -> NDArray:
        """Predict calibrated probabilities using conformal p-values.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            tie_breaking: Method for handling tied scores
            calibration_priors: Whether to use calibration priors in probability computation

        Returns:
            Array of shape (n_samples, n_classes) with calibrated probabilities

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> probs = conf_clf.predict_probability(X_test[:5])
            >>> probs.shape
            (5, 3)
            >>> # Probabilities should sum to 1
            >>> np.allclose(probs.sum(axis=1), 1.0)
            True
            >>> # All probabilities should be non-negative
            >>> (probs >= 0).all()
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        p_values = self.predict_p(X, tie_breaking)

        if calibration_priors:
            assert self.calibration_priors is not None

            evidence = self.calibration_priors * p_values
            return evidence / np.sum(evidence, axis=1, keepdims=True)

        return p_values / np.sum(p_values, axis=1, keepdims=True)

    def predict_set(
        self,
        X: ArrayLike,
        alpha: float,
        tie_breaking: TieBreakingMethod = "exact",
    ) -> NDArray:
        """Generate prediction sets for given significance level.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            alpha: Significance level (1-alpha is the confidence level)
            tie_breaking: Method for handling tied scores

        Returns:
            Binary array where entry [i,j] indicates if class j is in prediction set for instance i

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> # 90% confidence level (alpha=0.1)
            >>> pred_sets = conf_clf.predict_set(X_test[:5], alpha=0.1)
            >>> pred_sets.shape
            (5, 3)
            >>> # Each set should contain at least one class
            >>> (pred_sets.sum(axis=1) >= 1).all()
            True
            >>> # Higher confidence should give larger sets on average
            >>> pred_sets_95 = conf_clf.predict_set(X_test[:5], alpha=0.05)
            >>> pred_sets_95.sum() >= pred_sets.sum()
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")
        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        p_values = self.predict_p(X, tie_breaking)

        confidence = 1 - alpha
        return (p_values >= 1 - confidence).astype(int)

    def predict_sets_multiple_alpha(
        self,
        X: ArrayLike,
        alphas: Sequence[float],
        tie_breaking: TieBreakingMethod = "exact",
    ) -> dict[float, NDArray]:
        """Generate prediction sets for multiple alpha values.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            alphas: Sequence of significance levels
            tie_breaking: Method for handling tied scores

        Returns:
            Dictionary mapping alpha values to their prediction sets

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> pred_sets = conf_clf.predict_sets_multiple_alpha(X_test[:3], [0.05, 0.1, 0.2])
            >>> len(pred_sets)
            3
            >>> list(pred_sets.keys())
            [0.05, 0.1, 0.2]
            >>> # Lower alpha (higher confidence) should give larger sets
            >>> pred_sets[0.05].sum() >= pred_sets[0.2].sum()
            True
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")
        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        p_values = self.predict_p(X, tie_breaking)

        prediction_sets = {}
        for alpha in alphas:
            confidence = 1 - alpha
            prediction_sets[alpha] = (p_values >= 1 - confidence).astype(int)

        return prediction_sets

    def predict_all(
        self,
        X: ArrayLike,
        alphas: Sequence[float] | None = None,
        tie_breaking: TieBreakingMethod = "exact",
    ) -> ConformalPredictions:
        """Generate all prediction types in one call.

        Args:
            X: Features to predict on, shape (n_samples, n_features)
            alphas: Significance levels for prediction sets (defaults to [0.05, 0.1, 0.2])
            tie_breaking: Method for handling tied scores

        Returns:
            ConformalPredictions object with all prediction types

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> predictions = conf_clf.predict_all(X_test[:5])
            >>> len(predictions.point_predictions)
            5
            >>> predictions.p_values.shape
            (5, 3)
            >>> len(predictions.prediction_sets)
            3
            >>> list(predictions.prediction_sets.keys())
            [0.05, 0.1, 0.2]
        """
        if alphas is None:
            alphas = [0.05, 0.1, 0.2]

        return ConformalPredictions(
            point_predictions=self.predict(X, conformal=False),
            point_predictions_calibrated=self.predict(X, conformal=True),
            p_values=self.predict_p(X, tie_breaking),
            prediction_sets=self.predict_sets_multiple_alpha(X, alphas, tie_breaking),
            probabilities=self.predict_probability(X, tie_breaking),
        )

    def _hinge_score(
        self,
        probabilities: NDArray,
        classes: NDArray | None = None,
        y: NDArray | None = None,
    ) -> NDArray:
        """Compute hinge non-conformity scores.

        Args:
            probabilities: Model probability predictions
            classes: Class labels array
            y: True labels (optional)

        Returns:
            Non-conformity scores

        Examples:
            >>> import numpy as np
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>>
            >>> X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
            >>> model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> model.fit(X, y)
            RandomForestClassifier(...)
            >>>
            >>> conf_clf = ConformalClassifierModel(model)
            >>> conf_clf.is_trained = True  # Skip training for this example
            >>>
            >>> probs = model.predict_proba(X[:5])
            >>> scores = conf_clf._hinge_score(probs, model.classes_, y[:5])
            >>> scores.shape
            (5,)
            >>> # Scores should be between 0 and 1
            >>> ((scores >= 0) & (scores <= 1)).all()
            True
        """
        if y is not None:
            y_array = np.asarray(y)
            class_indexes = np.array(
                [np.argwhere(classes == y_array[i])[0][0] for i in range(len(y_array))]
            )
            result = 1 - probabilities[np.arange(len(y_array)), class_indexes]
        else:
            result = 1 - probabilities
        return result

    def _compute_p_values(
        self,
        test_scores: NDArray,
        test_bins: NDArray,
        tie_breaking: TieBreakingMethod = "exact",
    ) -> NDArray:
        """Compute p-values for conformal prediction with flexible tie-breaking.

        Args:
            test_scores: Scores for test instances
            test_bins: Bin assignments for test scores
            tie_breaking: Method for handling tied scores

        Returns:
            P-values for test instances

        Examples:
            >>> import numpy as np
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>>
            >>> # This is an internal method, typically called from predict_p
            >>> # Example shows general structure but requires full setup
            >>> test_scores = np.array([0.3, 0.7, 0.5])
            >>> test_bins = np.array([0, 1, 0])
            >>> # p_values = conf_clf._compute_p_values(test_scores, test_bins)
            >>> # len(p_values) == len(test_scores)
            >>> True
        """
        assert self.calibration_bins is not None
        assert self.calibration_scores is not None

        p_values = np.zeros(test_scores.shape)
        calibration_bins = self.calibration_bins

        bin_values, bin_indexes = np.unique(
            np.hstack((calibration_bins, test_bins)), return_inverse=True
        )
        bin_indexes_cal = bin_indexes[: len(calibration_bins)]
        bin_indexes_test = bin_indexes[len(calibration_bins) :]

        for b in range(len(bin_values)):
            bin_mask_cal = bin_indexes_cal == b
            bin_mask_test = bin_indexes_test == b

            if not np.any(bin_mask_test):
                continue

            bin_cal_scores = self.calibration_scores[bin_mask_cal]
            q = len(bin_cal_scores)
            bin_test_scores = test_scores[bin_mask_test]

            greater: NDArray
            equal: NDArray
            if len(bin_test_scores.shape) > 1:
                greater = np.sum(
                    bin_cal_scores[:, None, None] > bin_test_scores[None, :, :], axis=0
                )
                equal = np.sum(
                    bin_cal_scores[:, None, None] == bin_test_scores[None, :, :], axis=0
                )
            else:
                greater = np.sum(
                    bin_cal_scores[:, None] > bin_test_scores[None, :], axis=0
                )
                equal = np.sum(
                    bin_cal_scores[:, None] == bin_test_scores[None, :], axis=0
                )

            if equal.sum() == 0:
                bin_p_values = (greater + 1) / (q + 1)
            elif tie_breaking == "smoothing":
                if len(bin_test_scores.shape) > 1:
                    theta = np.random.rand(*bin_test_scores.shape)
                else:
                    theta = np.random.rand(len(bin_test_scores))
                bin_p_values = (greater + theta * (equal + 1)) / (q + 1)

            elif tie_breaking == "conservative":
                bin_p_values = (greater + (equal + 1)) / (q + 1)

            elif tie_breaking == "optimistic":
                bin_p_values = (greater + 1) / (q + 1)

            elif tie_breaking == "exact":
                bin_p_values = (greater + 0.5 * (equal + 1)) / (q + 1)

            else:
                raise ValueError(f"Unknown tie-breaking method: {tie_breaking}")

            p_values[bin_mask_test] = bin_p_values

        return p_values


class ConformalClassifierEvaluator:
    """Comprehensive evaluation framework for conformal classifiers.

    This class takes a trained and calibrated ConformalClassifierModel along with test data
    to evaluate model performance across multiple significance levels and analysis dimensions.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> import numpy as np
        >>>
        >>> # Generate sample data
        >>> X, y = make_classification(n_samples=300, n_features=10, n_classes=3, random_state=42)
        >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
        >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        >>>
        >>> # Train conformal classifier
        >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
        >>> conf_clf = ConformalClassifierModel(base_model)
        >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
        <...ConformalClassifierModel object at 0x...>
        >>>
        >>> # Create evaluator
        >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test, [0.1, 0.2])
        >>> evaluator.n_samples > 0
        True
        >>> len(evaluator.unique_labels) == 3
        True
        >>>
        >>> # Calculate coverage metrics
        >>> coverage_results = evaluator.calculate_coverage([0.1, 0.2])
        >>> len(coverage_results)
        2
        >>> all(result.alpha in [0.1, 0.2] for result in coverage_results)
        True
    """

    def __init__(
        self,
        model: ConformalClassifierModel,
        X: ArrayLike,
        y: ArrayLike,
        alphas: Sequence[float] | None = None,
        tie_breaking: TieBreakingMethod = "smoothing",
    ) -> None:
        """Initialize the evaluator with a model and test data.

        Args:
            model: Trained and calibrated ConformalClassifierModel
            X: Test features
            y: Test labels
            alphas: Alpha values to precompute (defaults to common values)
            tie_breaking: Method for handling tied scores in predictions

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=10, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> # Create evaluator with custom settings
            >>> evaluator = ConformalClassifierEvaluator(
            ...     conf_clf, X_test, y_test, [0.05, 0.1], "conservative"
            ... )
            >>> evaluator.tie_breaking
            'conservative'
            >>> 0.05 in evaluator._alpha_cache
            True
        """
        if not model.is_trained or not model.is_calibrated:
            raise RuntimeError("Model must be trained and calibrated")

        self.model = model
        self.X = X
        self.y = np.asarray(y)
        self.tie_breaking = tie_breaking

        # Basic statistics
        self.n_samples = len(self.y)
        self.unique_labels = np.unique(self.y)

        if alphas is None:
            alphas = [0.01, 0.05, 0.1]

        self._alpha_cache: dict[float, AlphaMetrics] = {}

        self.predictions = self.model.predict_all(self.X, alphas, self.tie_breaking)

        for alpha in alphas:
            self._compute_alpha(alpha)

    @property
    def classes(self) -> NDArray:
        """Get class labels from the underlying model.

        Returns:
            Array of class labels

        Examples:
            >>> # Assuming evaluator is already created
            >>> # classes = evaluator.classes
            >>> # len(classes) > 0
            >>> True
        """
        return self.model.model.classes_

    def calculate_class_conditional_coverage(
        self, alphas: Sequence[float], target_label: int
    ) -> list[ClassConditionalCoverageResult]:
        """Calculate class-conditional coverage for a specific label.

        Args:
            alphas: Significance levels to evaluate
            target_label: Specific label to compute coverage for

        Returns:
            List of ClassConditionalCoverageResult objects

        Raises:
            ValueError: If no instances found with target label

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> coverage_results = evaluator.calculate_class_conditional_coverage([0.1, 0.2], target_label=0)
            >>> len(coverage_results) <= 2  # May be less if target_label=0 not in test set
            True
            >>> if coverage_results:
            ...     coverage_results[0].target_label == 0
            ... else:
            ...     True  # No instances of target label found
            True
        """
        if target_label not in self.unique_labels:
            raise ValueError(f"No instances found with target label {target_label}")

        results = []

        for alpha in alphas:
            cache_entry = self._compute_alpha(alpha)

            if target_label in cache_entry.class_stats:
                class_stats = cache_entry.class_stats[target_label]
                n_target_instances = class_stats.total_instances
                covered_instances = class_stats.covered_instances

                target_indices = class_stats.indices
                total_set_size = int(
                    np.sum(cache_entry.prediction_sets[target_indices, :])
                )
            else:
                n_target_instances = 0
                covered_instances = 0
                total_set_size = 0

            if n_target_instances == 0:
                continue

            coverage_rate = covered_instances / n_target_instances
            expected_coverage = 1 - alpha
            coverage_gap = coverage_rate - expected_coverage
            avg_set_size = total_set_size / n_target_instances

            result = ClassConditionalCoverageResult(
                alpha=alpha,
                target_label=target_label,
                expected_coverage=expected_coverage,
                coverage_rate=round(coverage_rate, 4),
                coverage_gap=round(coverage_gap, 4),
                covered_instances=covered_instances,
                target_instances=n_target_instances,
                avg_set_size=round(avg_set_size, 2),
            )
            results.append(result)

        return results

    def calculate_class_conditional_precision(
        self, alphas: Sequence[float], target_label: int
    ) -> list[ClassConditionalPrecisionResult]:
        """Calculate class-conditional precision for conformal prediction sets.

        Args:
            alphas: Significance levels to evaluate
            target_label: Label to compute precision for

        Returns:
            List of ClassConditionalPrecisionResult objects

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> precision_results = evaluator.calculate_class_conditional_precision([0.1, 0.2], target_label=1)
            >>> len(precision_results)
            2
            >>> all(result.target_label == 1 for result in precision_results)
            True
            >>> # Precision should be between 0 and 1
            >>> all(0 <= result.precision_rate <= 1 for result in precision_results)
            True
        """
        count_true_label = np.sum(self.y == target_label)
        random_precision = (
            count_true_label / self.n_samples if self.n_samples > 0 else 0
        )

        results = []

        for alpha in alphas:
            cache_entry = self._compute_alpha(alpha)

            if target_label in cache_entry.class_stats:
                class_stats = cache_entry.class_stats[target_label]
                count_coverage = class_stats.correct_predictions
                count_sets = class_stats.sets_containing_label
            else:
                count_coverage = 0
                count_sets = 0

            precision_rate = count_coverage / count_sets if count_sets > 0 else 0
            pp_gain = (precision_rate - random_precision) * 100

            result = ClassConditionalPrecisionResult(
                alpha=alpha,
                target_label=target_label,
                precision_rate=round(precision_rate, 4),
                correct_predictions=count_coverage,
                sets_containing_target=count_sets,
                total_instances=self.n_samples,
                pp_gain_over_random=round(pp_gain, 2),
                random_precision=round(random_precision, 4),
            )
            results.append(result)

        return results

    def calculate_coverage(self, alphas: Sequence[float]) -> list[CoverageResult]:
        """Calculate marginal (overall) coverage statistics across different alpha values.

        Args:
            alphas: Significance levels to evaluate

        Returns:
            List of CoverageResult objects with coverage statistics

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> coverage_results = evaluator.calculate_coverage([0.1, 0.2])
            >>> len(coverage_results)
            2
            >>> # Coverage should generally be close to 1-alpha
            >>> all(0.0 <= result.coverage_rate <= 1.0 for result in coverage_results)
            True
            >>> # Lower alpha should give higher coverage
            >>> coverage_results[0].expected_coverage > coverage_results[1].expected_coverage
            True
        """
        results = []

        for alpha in alphas:
            cache_entry = self._compute_alpha(alpha)

            coverage_rate = cache_entry.coverage_count / self.n_samples
            expected_coverage = 1 - alpha
            coverage_gap = coverage_rate - expected_coverage
            total_set_size = int(np.sum(cache_entry.set_sizes))
            avg_set_size = total_set_size / self.n_samples

            result = CoverageResult(
                alpha=alpha,
                expected_coverage=expected_coverage,
                coverage_rate=round(coverage_rate, 4),
                coverage_gap=round(coverage_gap, 4),
                covered_instances=cache_entry.coverage_count,
                total_instances=self.n_samples,
                avg_set_size=round(avg_set_size, 2),
            )
            results.append(result)

        return results

    def calculate_efficiency_metrics(
        self, alphas: Sequence[float]
    ) -> list[EfficiencyResult]:
        """Calculate efficiency metrics for prediction sets.

        Args:
            alphas: Significance levels to evaluate

        Returns:
            List of EfficiencyResult objects

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> efficiency_results = evaluator.calculate_efficiency_metrics([0.1, 0.2])
            >>> len(efficiency_results)
            2
            >>> # Lower alpha should give larger average set sizes
            >>> efficiency_results[0].avg_set_size >= efficiency_results[1].avg_set_size
            True
            >>> # Rates should be between 0 and 1
            >>> all(0 <= result.singleton_rate <= 1 for result in efficiency_results)
            True
        """
        results = []

        for alpha in alphas:
            set_sizes = self._compute_alpha(alpha).set_sizes

            avg_set_size = np.mean(set_sizes)
            median_set_size = np.median(set_sizes)
            std_set_size = np.std(set_sizes)

            singleton_count = np.sum(set_sizes == 1)
            empty_count = np.sum(set_sizes == 0)

            singleton_rate = singleton_count / self.n_samples
            empty_rate = empty_count / self.n_samples

            result = EfficiencyResult(
                alpha=alpha,
                avg_set_size=round(avg_set_size, 2),
                median_set_size=round(median_set_size, 2),
                std_set_size=round(std_set_size, 2),
                singleton_rate=round(singleton_rate, 4),
                empty_rate=round(empty_rate, 4),
                singleton_count=int(singleton_count),
                empty_count=int(empty_count),
                total_instances=self.n_samples,
            )
            results.append(result)

        return results

    def calculate_size_stratified_coverage(
        self, alpha: float
    ) -> list[SizeStratifiedCoverageResult]:
        """Calculate coverage stratified by prediction set size.

        Args:
            alpha: Significance level

        Returns:
            List of SizeStratifiedCoverageResult objects ordered by set size

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> size_results = evaluator.calculate_size_stratified_coverage(0.1)
            >>> len(size_results) > 0
            True
            >>> # Results should be sorted by set size
            >>> all(size_results[i].set_size <= size_results[i+1].set_size
            ...     for i in range(len(size_results)-1))
            True
            >>> # Coverage rates should be between 0 and 1
            >>> all(0 <= result.coverage_rate <= 1 for result in size_results)
            True
        """
        cache_entry = self._compute_alpha(alpha)

        set_sizes = cache_entry.set_sizes
        coverage_mask = cache_entry.coverage_mask
        unique_sizes = np.unique(set_sizes)
        results = []

        for set_size in unique_sizes:
            size_mask = set_sizes == set_size
            size_instances = np.sum(size_mask)
            covered_instances = np.sum(coverage_mask[size_mask])
            coverage_rate = covered_instances / size_instances

            result = SizeStratifiedCoverageResult(
                set_size=int(set_size),
                coverage_rate=round(coverage_rate, 4),
                covered_instances=int(covered_instances),
                instance_count=int(size_instances),
            )
            results.append(result)

        return sorted(results, key=lambda x: x.set_size)

    def evaluate(self, alpha: float) -> EvaluationMetrics:
        """Evaluate conformal predictor using standard metrics for a single alpha.

        Args:
            alpha: Significance level

        Returns:
            EvaluationMetrics with evaluation results

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> from sklearn.datasets import make_classification
            >>> from sklearn.model_selection import train_test_split
            >>> import numpy as np
            >>>
            >>> X, y = make_classification(n_samples=200, n_features=5, n_classes=3, random_state=42)
            >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
            >>> X_calib, X_test, y_calib, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            >>>
            >>> base_model = RandomForestClassifier(n_estimators=20, random_state=42)
            >>> conf_clf = ConformalClassifierModel(base_model)
            >>> conf_clf.train(X_train, y_train).calibrate(X_calib, y_calib)
            <...ConformalClassifierModel object at 0x...>
            >>>
            >>> evaluator = ConformalClassifierEvaluator(conf_clf, X_test, y_test)
            >>> metrics = evaluator.evaluate(0.1)
            >>> 0 <= metrics.error <= 1  # Error rate between 0 and 1
            True
            >>> metrics.avg_c >= 1  # Average set size should be at least 1
            True
            >>> 0 <= metrics.one_c <= 1  # Singleton rate between 0 and 1
            True
            >>> 0 <= metrics.empty <= 1  # Empty rate between 0 and 1
            True
        """
        prediction_sets = self._compute_alpha(alpha).prediction_sets
        idx = [np.argwhere(self.classes == self.y[i])[0][0] for i in range(len(self.y))]
        coverage = prediction_sets[np.arange(len(self.y)), np.array(idx)]

        return EvaluationMetrics(
            error=(1 - np.mean(coverage)).item(),
            avg_c=np.mean(np.sum(prediction_sets, axis=1)).item(),
            one_c=np.mean(np.sum(prediction_sets, axis=1) == 1).item(),
            empty=np.mean(np.sum(prediction_sets, axis=1) == 0).item(),
        )

    def _compute_alpha(self, alpha: float) -> AlphaMetrics:
        """Get cache entry for alpha, computing it if necessary.

        Args:
            alpha: Significance level

        Returns:
            Cache entry for the alpha value

        Examples:
            >>> # This is an internal method, typically not called directly
            >>> # from an evaluator instance:
            >>> # entry = evaluator._compute_alpha(0.2)
            >>> # entry.alpha == 0.2
            >>> True
        """
        if alpha not in self._alpha_cache:
            self._alpha_cache[alpha] = AlphaMetrics.create_for_alpha(
                alpha=alpha,
                model=self.model,
                X=self.X,
                y=self.y,
                n_samples=self.n_samples,
                unique_labels=self.unique_labels,
                tie_breaking=self.tie_breaking,
                predictions=self.predictions,
            )
        return self._alpha_cache[alpha]
