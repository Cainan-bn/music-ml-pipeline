
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from playcatch.config.logging_config import get_logger
from playcatch.domain.interfaces import IProbabilisticClassifier, IValidationStrategy

logger = get_logger(__name__)

ModelFactoryFn = Callable[[], IProbabilisticClassifier]


class ModelEvaluator:
    """Avalia um modelo probabilístico usando uma estratégia de validação plugável."""

    def __init__(self, validation_strategy: IValidationStrategy) -> None:
            self._validation_strategy = validation_strategy

    def evaluate(
        self,
        model_factory: ModelFactoryFn,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> dict[str, object]:
            fold_accuracies: list[float] = []
        classification_reports: list[str] = []
        confusion_matrices: list[np.ndarray] = []

        splits = self._validation_strategy.split(features, labels)

        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            x_train, x_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

            model = model_factory()
            model.train(x_train, y_train)
            proba = model.predict_proba(x_val)
            predicted_labels = proba.idxmax(axis=1)

            accuracy = accuracy_score(y_val, predicted_labels)
            fold_accuracies.append(accuracy)
            classification_reports.append(
                classification_report(y_val, predicted_labels, zero_division=0)
            )
            confusion_matrices.append(confusion_matrix(y_val, predicted_labels))

            logger.info("Fold %d: accuracy=%.3f", fold_idx, accuracy)

        mean_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
        logger.info("Acurácia média (%d folds): %.3f", len(fold_accuracies), mean_accuracy)

        return {
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": mean_accuracy,
            "classification_reports": classification_reports,
            "confusion_matrices": confusion_matrices,
        }