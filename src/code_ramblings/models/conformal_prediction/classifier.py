"""
Self-contained ConformalClassifier integrating custom conformal prediction logic.

This implementation combines the comprehensive evaluation framework from "mine"
with the core conformal prediction algorithms from "other", eliminating all
external dependencies.
"""

from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.stats import kstest  # type: ignore[import-untyped]


class SklearnClassifier(Protocol):
    """Protocol for sklearn-compatible classifiers used in conformal prediction."""

    classes_: NDArray

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the classifier to training data."""
        ...

    def predict(self, X: pd.DataFrame) -> NDArray:
        """Predict class labels for samples."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> NDArray:
        """Predict class probabilities for samples."""
        ...


class ConformalClassifier:
    """
    A conformal prediction classifier with comprehensive evaluation metrics.

    This class wraps sklearn-compatible classifiers to provide conformal prediction
    capabilities along with detailed coverage and efficiency analysis.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> base_model = RandomForestClassifier()
        >>> conf_clf = ConformalClassifier(base_model)
        >>> conf_clf.train(X_train, y_train).calib(X_calib, y_calib)
        >>> predictions = conf_clf.predict(X_test)
        >>> coverage_df = conf_clf.calculate_coverage(X_test, y_test, [0.05, 0.1, 0.2])
    """

    def __init__(self, model: SklearnClassifier) -> None:
        """
        Initialize the conformal classifier.

        Args:
            model: sklearn-compatible classifier

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> conf_clf = ConformalClassifier(RandomForestClassifier())
        """
        self.model = model

        # Conformal prediction state
        self.calibration_scores: NDArray | None = None
        self.calibration_bins: pd.Series | None = None
        self.class_conditional: bool = False

        # Training state
        self.is_trained = False
        self.is_calibrated = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """
        Train the base classifier.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining

        Examples:
            >>> conf_clf = ConformalClassifier(RandomForestClassifier())
            >>> conf_clf.train(X_train, y_train)
            <__main__.ConformalClassifier object at 0x...>
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def calibrate(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Calibrate the conformal predictor using the calibration set.

        Args:
            X: Calibration features
            y: Calibration labels

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If model hasn't been trained yet

        Examples:
            >>> conf_clf.calibrate(X_calib, y_calib)
            <__main__.ConformalClassifier object at 0x...>
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        probabilities = self.model.predict_proba(X)
        self.calibration_scores = self._hinge_score(
            probabilities, self.model.classes_, y
        )
        self.calibration_bins = y

        self.is_calibrated = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make point predictions using the base classifier.

        Args:
            X: Features to predict on

        Returns:
            Series of predicted labels

        Raises:
            RuntimeError: If model hasn't been trained yet

        Examples:
            >>> predictions = conf_clf.predict(X_test)
            >>> print(predictions.head())
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")

        return pd.Series(self.model.predict(X), index=X.index, name="prediction")

    def predict_set(self, X: pd.DataFrame, alpha: float) -> NDArray:
        """
        Generate prediction sets for given significance level.

        Args:
            X: Features to predict on
            alpha: Significance level (1-alpha is the confidence level)

        Returns:
            Binary array where entry [i,j] indicates if class j is in prediction set for instance i

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
            >>> print(f"Average set size: {np.mean(np.sum(pred_sets, axis=1)):.2f}")
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")
        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        # Get non-conformity scores for test set
        probabilities = self.model.predict_proba(X)
        test_scores = self._hinge_score(probabilities)  # All classes

        confidence = 1 - alpha
        prediction_sets = []

        # Generate prediction set for each class
        for c in range(len(self.model.classes_)):
            assert self.calibration_scores is not None
            bins_test = np.full(len(X), self.model.classes_[c])
            p_values = self._compute_p_values(
                self.calibration_scores,
                test_scores[:, c],
                self.calibration_bins,
                bins_test,
            )

            # Include class c if p-value >= 1-confidence
            class_predictions = (p_values >= 1 - confidence).astype(int)
            prediction_sets.append(class_predictions)

        return np.array(prediction_sets).T

    def evaluate(self, X: pd.DataFrame, y: pd.Series, alpha: float) -> dict[str, float]:
        """
        Evaluate conformal predictor using standard metrics.

        Args:
            X: Test features
            y: Test labels
            alpha: Significance level

        Returns:
            Dictionary with evaluation metrics including error, coverage, avg set size, etc.

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> results = conf_clf.evaluate(X_test, y_test, alpha=0.1)
            >>> print(f"Coverage: {1 - results['error']:.3f}")
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")
        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        # Generate prediction sets
        prediction_sets = self.predict_set(X, alpha)

        # Compute p-values for KS test
        probabilities = self.model.predict_proba(X)
        test_scores = self._hinge_score(probabilities)

        p_values_for_ks = []
        for i, true_class in enumerate(y):
            class_idx = np.argwhere(self.model.classes_ == true_class)[0][0]
            assert self.calibration_scores is not None
            bins_test = np.array([true_class])
            p_val = self._compute_p_values(
                self.calibration_scores,
                test_scores[i : i + 1, class_idx],
                self.calibration_bins,
                bins_test,
            )[0]

            p_values_for_ks.append(p_val)

        p_values_array = np.array(p_values_for_ks)

        # Calculate metrics using the integrated function
        metrics = ["error", "avg_c", "one_c", "empty", "ks_test"]
        results = self._get_classification_results(
            prediction_sets, p_values_array, self.model.classes_, y.values, metrics
        )

        return results

    def calculate_coverage(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float]
    ) -> pd.DataFrame:
        """
        Calculate marginal (overall) coverage statistics across different alpha values.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate

        Returns:
            DataFrame with coverage statistics indexed by alpha

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> coverage_df = conf_clf.calculate_coverage(X_test, y_test, [0.05, 0.1, 0.2])
            >>> print(coverage_df['coverage_rate'])
            alpha
            0.05    0.95
            0.10    0.90
            0.20    0.80
            Name: coverage_rate, dtype: float64
        """
        base_metrics = self._compute_base_metrics(X, y, alphas)
        n_samples = base_metrics["n_samples"]

        results = []

        for alpha in alphas:
            alpha_stats = base_metrics["alpha_stats"][alpha]

            coverage_count = alpha_stats["coverage_count"]
            total_set_size = alpha_stats["total_set_size"]

            coverage_rate = coverage_count / n_samples
            expected_coverage = 1 - alpha
            coverage_gap = coverage_rate - expected_coverage
            avg_set_size = total_set_size / n_samples

            result = {
                "alpha": alpha,
                "expected_coverage": expected_coverage,
                "coverage_rate": round(coverage_rate, 4),
                "coverage_gap": round(coverage_gap, 4),
                "covered_instances": coverage_count,
                "total_instances": n_samples,
                "avg_set_size": round(avg_set_size, 2),
            }
            results.append(result)

        return pd.DataFrame(results).set_index("alpha")

    def calculate_class_conditional_coverage(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float], target_label: int
    ) -> pd.DataFrame:
        """
        Calculate class-conditional coverage for a specific label.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate
            target_label: Specific label to compute coverage for

        Returns:
            DataFrame with class-conditional coverage statistics

        Raises:
            RuntimeError: If model hasn't been trained or calibrated
            ValueError: If no instances found with target label

        Examples:
            >>> coverage_df = conf_clf.calculate_class_conditional_coverage(
            ...     X_test, y_test, [0.1, 0.2], target_label=0
            ... )
            >>> print(coverage_df['coverage_rate'])
            alpha
            0.1     0.88
            0.2     0.82
            Name: coverage_rate, dtype: float64
        """
        base_metrics = self._compute_base_metrics(X, y, alphas)

        # Check if target label exists
        if target_label not in base_metrics["unique_labels"]:
            raise ValueError(f"No instances found with target label {target_label}")

        results = []

        for alpha in alphas:
            alpha_stats = base_metrics["alpha_stats"][alpha]
            class_stats = alpha_stats["class_stats"][target_label]

            n_target_instances = class_stats["total_instances"]
            covered_instances = class_stats["covered_instances"]

            # Calculate average set size for target label instances
            target_indices = class_stats["indices"]
            sets = alpha_stats["prediction_sets"]
            total_set_size = sum(np.sum(sets[i, :]) for i in target_indices)

            coverage_rate = covered_instances / n_target_instances
            expected_coverage = 1 - alpha
            coverage_gap = coverage_rate - expected_coverage
            avg_set_size = total_set_size / n_target_instances

            result = {
                "alpha": alpha,
                "target_label": target_label,
                "expected_coverage": expected_coverage,
                "coverage_rate": round(coverage_rate, 4),
                "coverage_gap": round(coverage_gap, 4),
                "covered_instances": covered_instances,
                "target_instances": n_target_instances,
                "avg_set_size": round(avg_set_size, 2),
            }
            results.append(result)

        return pd.DataFrame(results).set_index("alpha")

    def calculate_class_conditional_precision(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float], target_label: int
    ) -> pd.DataFrame:
        """
        Calculate class-conditional precision: of prediction sets containing target_label,
        what fraction have true label = target_label.

        This is a precision-like measure for conformal prediction sets.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate
            target_label: Label to compute precision for

        Returns:
            DataFrame with precision statistics indexed by alpha

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> precision_df = conf_clf.calculate_class_conditional_precision(
            ...     X_test, y_test, [0.1, 0.2], target_label=1
            ... )
            >>> print(precision_df['precision_rate'])
            alpha
            0.1     0.85
            0.2     0.82
            Name: precision_rate, dtype: float64
        """
        base_metrics = self._compute_base_metrics(X, y, alphas)
        n_samples = base_metrics["n_samples"]

        # Count instances with true target label for baseline
        count_true_label = (
            sum(1 for label in base_metrics["unique_labels"] if label == target_label)
            if target_label in base_metrics["unique_labels"]
            else 0
        )

        # Get actual count from class stats if available
        if target_label in base_metrics["alpha_stats"][alphas[0]]["class_stats"]:
            count_true_label = base_metrics["alpha_stats"][alphas[0]]["class_stats"][
                target_label
            ]["total_instances"]

        random_precision = count_true_label / n_samples if n_samples > 0 else 0

        results = []

        for alpha in alphas:
            alpha_stats = base_metrics["alpha_stats"][alpha]

            # Get precision statistics for target label
            if target_label in alpha_stats["class_stats"]:
                class_stats = alpha_stats["class_stats"][target_label]
                count_coverage = class_stats["correct_predictions"]
                count_sets = class_stats["sets_containing_label"]
            else:
                count_coverage = 0
                count_sets = 0

            precision_rate = count_coverage / count_sets if count_sets > 0 else 0
            pp_gain = (precision_rate - random_precision) * 100

            result = {
                "alpha": alpha,
                "target_label": target_label,
                "precision_rate": round(precision_rate, 4),
                "correct_predictions": count_coverage,
                "sets_containing_target": count_sets,
                "total_instances": n_samples,
                "pp_gain_over_random": round(pp_gain, 2),
                "random_precision": round(random_precision, 4),
            }
            results.append(result)

        return pd.DataFrame(results).set_index("alpha")

    def calculate_efficiency_metrics(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float]
    ) -> pd.DataFrame:
        """
        Calculate efficiency metrics for prediction sets.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate

        Returns:
            DataFrame with efficiency metrics indexed by alpha

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> efficiency_df = conf_clf.calculate_efficiency_metrics(
            ...     X_test, y_test, [0.05, 0.1, 0.2]
            ... )
            >>> print(efficiency_df[['avg_set_size', 'singleton_rate']])
                     avg_set_size  singleton_rate
            alpha
            0.05             2.3            0.45
            0.10             1.8            0.60
            0.20             1.4            0.75
        """
        base_metrics = self._compute_base_metrics(X, y, alphas)
        n_samples = base_metrics["n_samples"]

        results = []

        for alpha in alphas:
            alpha_stats = base_metrics["alpha_stats"][alpha]
            set_sizes = alpha_stats["set_sizes"]

            # Efficiency metrics
            avg_set_size = np.mean(set_sizes)
            median_set_size = np.median(set_sizes)
            std_set_size = np.std(set_sizes)

            # Special cases
            singleton_count = sum(1 for size in set_sizes if size == 1)
            empty_count = sum(1 for size in set_sizes if size == 0)

            singleton_rate = singleton_count / n_samples
            empty_rate = empty_count / n_samples

            result = {
                "alpha": alpha,
                "avg_set_size": round(avg_set_size, 2),
                "median_set_size": round(median_set_size, 2),
                "std_set_size": round(std_set_size, 2),
                "singleton_rate": round(singleton_rate, 4),
                "empty_rate": round(empty_rate, 4),
                "singleton_count": singleton_count,
                "empty_count": empty_count,
                "total_instances": n_samples,
            }
            results.append(result)

        return pd.DataFrame(results).set_index("alpha")

    def calculate_size_stratified_coverage(
        self, X: pd.DataFrame, y: pd.Series, alpha: float
    ) -> pd.DataFrame:
        """
        Calculate coverage stratified by prediction set size.

        This shows how coverage varies depending on the size of prediction sets,
        which can reveal if the conformal predictor is well-calibrated across
        different uncertainty levels.

        Args:
            X: Test features
            y: Test labels
            alpha: Significance level

        Returns:
            DataFrame with coverage by set size

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> size_cov_df = conf_clf.calculate_size_stratified_coverage(
            ...     X_test, y_test, 0.1
            ... )
            >>> print(size_cov_df)
                 coverage_rate  instance_count
            set_size
            1              0.95              45
            2              0.88              32
            3              0.92              23
        """
        if not self.is_trained or not self.is_calibrated:
            raise RuntimeError("Model must be trained and calibrated")

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        n_samples = len(y)

        sets = self.predict_set(X, alpha)

        # Group instances by set size
        size_to_instances: dict[int, list[int]] = {}
        for i in range(n_samples):
            set_size = int(np.sum(sets[i, :]))
            if set_size not in size_to_instances:
                size_to_instances[set_size] = []
            size_to_instances[set_size].append(i)

        results = []

        for set_size, instances in size_to_instances.items():
            covered_count = sum(1 for i in instances if sets[i, y.iloc[i]])
            coverage_rate = covered_count / len(instances)

            result = {
                "set_size": set_size,
                "coverage_rate": round(coverage_rate, 4),
                "covered_instances": covered_count,
                "instance_count": len(instances),
            }
            results.append(result)

        df = pd.DataFrame(results).set_index("set_size")
        return df.sort_index()

    def comprehensive_evaluation(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float]
    ) -> dict[str, pd.DataFrame]:
        """
        Run all evaluation metrics and return comprehensive results.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate

        Returns:
            Dictionary containing all evaluation DataFrames

        Raises:
            RuntimeError: If model hasn't been trained or calibrated

        Examples:
            >>> results = conf_clf.comprehensive_evaluation(X_test, y_test, [0.05, 0.1])
            >>> print(results['coverage']['coverage_rate'])
            alpha
            0.05    0.95
            0.10    0.90
            Name: coverage_rate, dtype: float64
        """
        if not self.is_trained or not self.is_calibrated:
            raise RuntimeError("Model must be trained and calibrated")

        results = {
            "coverage": self.calculate_coverage(X, y, alphas),
            "efficiency": self.calculate_efficiency_metrics(X, y, alphas),
        }

        # Add size-stratified coverage for the middle alpha value
        if alphas:
            mid_alpha = alphas[len(alphas) // 2]
            results["size_stratified_coverage"] = (
                self.calculate_size_stratified_coverage(X, y, mid_alpha)
            )

        return results

    def _hinge_score(
        self,
        probabilities: NDArray,
        classes: NDArray | None = None,
        y: pd.Series | None = None,
    ) -> NDArray:
        """
        Compute hinge non-conformity scores.

        Args:
            probabilities: Predicted class probabilities
            classes: Class labels (needed when y is provided)
            y: True labels (if provided, returns scores for true classes only)

        Returns:
            Non-conformity scores
        """
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            class_indexes = np.array(
                [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
            )
            result = 1 - probabilities[np.arange(len(y)), class_indexes]
        else:
            result = 1 - probabilities
        return result

    def _compute_p_values(
        self,
        calibration_scores: NDArray,
        test_scores: NDArray,
        calibration_bins: pd.Series | None = None,
        test_bins: NDArray | None = None,
        smoothing: bool = True,
    ) -> NDArray:
        """
        Compute p-values for conformal prediction.

        Args:
            calibration_scores: Non-conformity scores from calibration set
            test_scores: Non-conformity scores for test instances
            calibration_bins: Mondrian categories for calibration set
            test_bins: Mondrian categories for test instances
            smoothing: Whether to use smoothed p-values

        Returns:
            P-values for test instances
        """
        p_values = np.zeros(test_scores.shape)

        if calibration_bins is None:
            # Marginal conformal prediction
            q = len(calibration_scores)
            if smoothing:
                if len(test_scores.shape) > 1:
                    thetas = np.random.rand(test_scores.shape[0], test_scores.shape[1])
                    p_values = np.array(
                        [
                            [
                                (
                                    np.sum(calibration_scores > test_scores[i, c])
                                    + thetas[i, c]
                                    * (
                                        np.sum(calibration_scores == test_scores[i, c])
                                        + 1
                                    )
                                )
                                / (q + 1)
                                for c in range(test_scores.shape[1])
                            ]
                            for i in range(len(test_scores))
                        ]
                    )
                else:
                    thetas = np.random.rand(len(test_scores))
                    p_values = np.array(
                        [
                            (
                                np.sum(calibration_scores > test_scores[i])
                                + thetas[i]
                                * (np.sum(calibration_scores == test_scores[i]) + 1)
                            )
                            / (q + 1)
                            for i in range(len(test_scores))
                        ]
                    )
            else:
                if len(test_scores.shape) > 1:
                    p_values = np.array(
                        [
                            [
                                (np.sum(calibration_scores >= test_scores[i, c]) + 1)
                                / (q + 1)
                                for c in range(test_scores.shape[1])
                            ]
                            for i in range(len(test_scores))
                        ]
                    )
                else:
                    p_values = np.array(
                        [
                            (np.sum(calibration_scores >= test_scores[i]) + 1) / (q + 1)
                            for i in range(len(test_scores))
                        ]
                    )
        else:
            assert test_bins is not None
            # Class-conditional (Mondrian) conformal prediction
            if isinstance(calibration_bins, pd.Series):
                calibration_bins = calibration_bins.values

            bin_values, bin_indexes = np.unique(
                np.hstack((calibration_bins, test_bins)), return_inverse=True
            )
            bin_indexes_cal = bin_indexes[: len(calibration_bins)]
            bin_indexes_test = bin_indexes[len(calibration_bins) :]

            for b in range(len(bin_values)):
                bin_cal_scores = calibration_scores[bin_indexes_cal == b]
                q = len(bin_cal_scores)
                bin_test_scores = test_scores[bin_indexes_test == b]

                if len(bin_test_scores) == 0:
                    continue

                if smoothing:
                    if len(bin_test_scores.shape) > 1:
                        thetas = np.random.rand(
                            bin_test_scores.shape[0], bin_test_scores.shape[1]
                        )
                        bin_p_values = np.array(
                            [
                                [
                                    (
                                        np.sum(bin_cal_scores > bin_test_scores[i, c])
                                        + thetas[i, c]
                                        * (
                                            np.sum(
                                                bin_cal_scores == bin_test_scores[i, c]
                                            )
                                            + 1
                                        )
                                    )
                                    / (q + 1)
                                    for c in range(test_scores.shape[1])
                                ]
                                for i in range(len(bin_test_scores))
                            ]
                        )
                    else:
                        thetas = np.random.rand(len(bin_test_scores))
                        bin_p_values = np.array(
                            [
                                (
                                    np.sum(bin_cal_scores > bin_test_scores[i])
                                    + thetas[i]
                                    * (np.sum(bin_cal_scores == bin_test_scores[i]) + 1)
                                )
                                / (q + 1)
                                for i in range(len(bin_test_scores))
                            ]
                        )
                else:
                    if len(bin_test_scores.shape) > 1:
                        bin_p_values = np.array(
                            [
                                [
                                    (
                                        np.sum(bin_cal_scores >= bin_test_scores[i, c])
                                        + 1
                                    )
                                    / (q + 1)
                                    for c in range(test_scores.shape[1])
                                ]
                                for i in range(len(bin_test_scores))
                            ]
                        )
                    else:
                        bin_p_values = np.array(
                            [
                                (np.sum(bin_cal_scores >= bin_test_scores[i]) + 1)
                                / (q + 1)
                                for i in range(len(bin_test_scores))
                            ]
                        )

                orig_indexes = np.arange(len(test_scores))[bin_indexes_test == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values

        return p_values

    def _get_classification_results(
        self,
        prediction_sets: NDArray,
        p_values: NDArray,
        classes: NDArray,
        y: NDArray,
        metrics: list[str],
    ) -> dict[str, float]:
        """
        Calculate classification results for conformal prediction evaluation.

        Args:
            prediction_sets: Binary prediction sets
            p_values: P-values for true classes
            classes: Class labels
            y: True labels
            metrics: List of metrics to compute

        Returns:
            Dictionary with computed metrics
        """
        test_results = {}
        class_indexes = np.array(
            [np.argwhere(classes == y[i])[0][0] for i in range(len(y))]
        )

        if "error" in metrics:
            test_results["error"] = 1 - np.sum(
                prediction_sets[np.arange(len(y)), class_indexes]
            ) / len(y)
        if "avg_c" in metrics:
            test_results["avg_c"] = np.sum(prediction_sets) / len(y)
        if "one_c" in metrics:
            test_results["one_c"] = np.sum(
                [np.sum(p) == 1 for p in prediction_sets]
            ) / len(y)
        if "empty" in metrics:
            test_results["empty"] = np.sum(
                [np.sum(p) == 0 for p in prediction_sets]
            ) / len(y)
        if "ks_test" in metrics:
            test_results["ks_test"] = kstest(p_values, "uniform").pvalue

        return test_results

    def _compute_prediction_sets(
        self, X: pd.DataFrame, alphas: Sequence[float]
    ) -> dict[float, NDArray]:
        """
        Compute prediction sets once for all alphas to avoid redundant computation.

        Args:
            X: Features to predict on
            alphas: Significance levels to evaluate

        Returns:
            Dictionary mapping alpha values to their prediction sets

        Raises:
            RuntimeError: If model hasn't been trained or calibrated
        """
        if not self.is_trained:
            raise RuntimeError("Model isn't trained")
        if not self.is_calibrated:
            raise RuntimeError("Model isn't calibrated")

        return {alpha: self.predict_set(X, alpha) for alpha in alphas}

    def _compute_base_metrics(
        self, X: pd.DataFrame, y: pd.Series, alphas: Sequence[float]
    ) -> dict:
        """
        Compute all basic statistics needed by evaluation methods.

        This method centralizes common computations to avoid redundancy across
        different evaluation methods.

        Args:
            X: Test features
            y: Test labels
            alphas: Significance levels to evaluate

        Returns:
            Dictionary containing base metrics and preprocessed data

        Raises:
            RuntimeError: If model hasn't been trained or calibrated
        """
        if not self.is_trained or not self.is_calibrated:
            raise RuntimeError("Model must be trained and calibrated")

        # Preprocess data once
        X_clean = X.reset_index(drop=True)
        y_clean = y.reset_index(drop=True)
        n_samples = len(y_clean)

        # Compute prediction sets once for all alphas
        prediction_sets = self._compute_prediction_sets(X_clean, alphas)

        # Compute basic statistics for each alpha
        base_stats = {}

        for alpha in alphas:
            sets = prediction_sets[alpha]

            # Coverage statistics
            covered_instances = [
                i for i in range(n_samples) if sets[i, y_clean.iloc[i]]
            ]
            coverage_count = len(covered_instances)

            # Set size statistics
            set_sizes = [np.sum(sets[i, :]) for i in range(n_samples)]

            # Per-class statistics
            unique_labels = np.unique(y_clean)
            class_stats = {}

            for label in unique_labels:
                label_indices = [
                    i for i in range(n_samples) if y_clean.iloc[i] == label
                ]

                # Class-conditional coverage
                class_covered = sum(1 for i in label_indices if sets[i, label])

                # Class-conditional precision (sets containing this label)
                sets_with_label = [i for i in range(n_samples) if sets[i, label]]
                correct_predictions = sum(
                    1 for i in sets_with_label if y_clean.iloc[i] == label
                )

                class_stats[label] = {
                    "total_instances": len(label_indices),
                    "covered_instances": class_covered,
                    "sets_containing_label": len(sets_with_label),
                    "correct_predictions": correct_predictions,
                    "indices": label_indices,
                    "sets_with_label_indices": sets_with_label,
                }

            base_stats[alpha] = {
                "prediction_sets": sets,
                "coverage_count": coverage_count,
                "covered_indices": covered_instances,
                "set_sizes": set_sizes,
                "class_stats": class_stats,
                "total_set_size": np.sum(sets),
            }

        return {
            "X_clean": X_clean,
            "y_clean": y_clean,
            "n_samples": n_samples,
            "unique_labels": unique_labels,
            "alpha_stats": base_stats,
        }
