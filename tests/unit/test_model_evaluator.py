
import pandas as pd
import pytest

from playcatch.application.evaluation.model_evaluator import ModelEvaluator
from playcatch.application.strategies.validation_strategies import (
    KFoldValidationStrategy,
    TrainTestSplitStrategy,
)
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender


@pytest.fixture
def synthetic_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "feat_a": [1.0, 0.0] * 10,
            "feat_b": [0.0, 1.0] * 10,
        }
    )
    labels = pd.Series(["happy", "sad"] * 10)
    return features, labels


class TestModelEvaluator:

    def test_evaluate_with_kfold_returns_metrics_per_fold(
        self, synthetic_features_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, labels = synthetic_features_and_labels
        evaluator = ModelEvaluator(validation_strategy=KFoldValidationStrategy(n_splits=4))

        results = evaluator.evaluate(
            model_factory=lambda: LightGbmRecommender(num_boost_round=10),
            features=features,
            labels=labels,
        )

        assert len(results["fold_accuracies"]) == 4
        assert 0.0 <= results["mean_accuracy"] <= 1.0
        assert len(results["classification_reports"]) == 4

    def test_evaluate_with_train_test_split_returns_single_fold(
        self, synthetic_features_and_labels: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        features, labels = synthetic_features_and_labels
        evaluator = ModelEvaluator(
            validation_strategy=TrainTestSplitStrategy(test_size=0.3)
        )

        results = evaluator.evaluate(
            model_factory=lambda: LightGbmRecommender(num_boost_round=10),
            features=features,
            labels=labels,
        )

        assert len(results["fold_accuracies"]) == 1