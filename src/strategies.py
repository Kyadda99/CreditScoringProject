"""Interchangeable estimator and imbalance-handling variants.

The two axes the comparison grid varies: estimator family and how the rare
class is handled. Class weighting is expressed differently in scikit-learn and
XGBoost, so each estimator family knows its own form rather than the caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import RANDOM_STATE


class EstimatorStrategy(ABC):
    """Estimator family.

    Contract: `build` returns a new, unfitted estimator on every call and
    decides how to express class weighting in its own API.

    Attributes:
        name: Key in `ESTIMATOR_STRATEGIES`, used in the comparison table.
    """

    name: str

    @abstractmethod
    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Create the estimator.

        Args:
            class_weighted: Whether the rare class gets a higher weight.
            scale_pos_weight: Ratio of negatives to positives; relevant only
                for families that express weights as a number.

        Returns:
            An unfitted estimator.
        """


class ImbalanceStrategy(ABC):
    """How class imbalance is handled.

    Contract: an arm either asks the estimator for weights
    (`uses_class_weights`) or adds a sampler (`make_sampler`), never both, or
    the same imbalance would be counted twice.

    Attributes:
        name: Key in `IMBALANCE_STRATEGIES`.
        uses_class_weights: Whether the estimator gets `class_weighted=True`.
    """

    name: str
    uses_class_weights: bool = False

    @abstractmethod
    def make_sampler(self) -> BaseEstimator | None:
        """Return a fresh sampler, or `None` when the arm uses none."""


@dataclass(frozen=True)
class LogisticRegressionStrategy(EstimatorStrategy):
    """Logistic regression: the project baseline."""

    name: str = "logistic_regression"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Create the logistic regression. See `EstimatorStrategy.build`."""
        # max_iter=1000: one-hot gives 211 columns and lbfgs does not converge in 100.
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced" if class_weighted else None,
        )


@dataclass(frozen=True)
class RandomForestStrategy(EstimatorStrategy):
    """Random forest with bounded depth."""

    name: str = "random_forest"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Create the random forest. See `EstimatorStrategy.build`."""
        # Depth and leaf limits: without them the forest on 246k rows trains
        # slowly, weighs hundreds of MB and overfits rare-class noise.
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced" if class_weighted else None,
        )


@dataclass(frozen=True)
class XGBoostStrategy(EstimatorStrategy):
    """Gradient boosting; expresses class weights as a number, not a label."""

    name: str = "xgboost"

    def build(self, *, class_weighted: bool, scale_pos_weight: float) -> BaseEstimator:
        """Create the XGBoost model. See `EstimatorStrategy.build`."""
        # tree_method="hist": histogram splits, much faster on 246k x 211.
        # eval_metric="aucpr" matches the metric used to pick the champion.
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight if class_weighted else 1.0,
        )


@dataclass(frozen=True)
class NoImbalanceHandling(ImbalanceStrategy):
    """No correction: the model sees the distribution as it is."""

    name: str = "none"
    uses_class_weights: bool = False

    def make_sampler(self) -> BaseEstimator | None:
        """Uses no sampler."""
        return None


@dataclass(frozen=True)
class ClassWeighting(ImbalanceStrategy):
    """Higher weight for the rare class, expressed in the estimator API."""

    name: str = "class_weight"
    uses_class_weights: bool = True

    def make_sampler(self) -> BaseEstimator | None:
        """Uses no sampler."""
        return None


@dataclass(frozen=True)
class Resampling(ImbalanceStrategy):
    """Synthetic rare-class examples (SMOTE)."""

    name: str = "resample"
    uses_class_weights: bool = False

    def make_sampler(self) -> BaseEstimator | None:
        """Return a fresh SMOTE."""
        # No n_jobs: the parameter was removed from SMOTE in imbalanced-learn 0.12.
        return SMOTE(random_state=RANDOM_STATE)


ESTIMATOR_STRATEGIES: dict[str, EstimatorStrategy] = {
    s.name: s
    for s in (
        LogisticRegressionStrategy(),
        RandomForestStrategy(),
        XGBoostStrategy(),
    )
}
"""Algorithm name -> strategy. A new algorithm adds an entry, not a branch."""

IMBALANCE_STRATEGIES: dict[str, ImbalanceStrategy] = {
    s.name: s for s in (NoImbalanceHandling(), ClassWeighting(), Resampling())
}
"""Arm name -> strategy."""


def get_estimator_strategy(name: str) -> EstimatorStrategy:
    """Return the estimator strategy with the given name.

    Args:
        name: Key from `ESTIMATOR_STRATEGIES`.

    Returns:
        The estimator strategy.

    Raises:
        ValueError: When the name is unknown.
    """
    try:
        return ESTIMATOR_STRATEGIES[name]
    except KeyError:
        allowed = ", ".join(sorted(ESTIMATOR_STRATEGIES))
        raise ValueError(f"Unknown estimator: {name!r}. Allowed: {allowed}.") from None


def get_imbalance_strategy(name: str) -> ImbalanceStrategy:
    """Return the imbalance strategy with the given name.

    Args:
        name: Key from `IMBALANCE_STRATEGIES`.

    Returns:
        The imbalance strategy.

    Raises:
        ValueError: When the name is unknown.
    """
    try:
        return IMBALANCE_STRATEGIES[name]
    except KeyError:
        allowed = ", ".join(sorted(IMBALANCE_STRATEGIES))
        raise ValueError(
            f"Unknown imbalance arm: {name!r}. Allowed: {allowed}."
        ) from None
