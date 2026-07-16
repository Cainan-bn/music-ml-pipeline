
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from playcatch.domain.exceptions import PlaycatchDomainError
from playcatch.domain.interfaces import IValidationStrategy


class KFoldValidationStrategy(IValidationStrategy):
    """Validação cruzada em K dobras."""

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
            self._n_splits = n_splits
        self._random_state = random_state

    def split(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            if len(features) < self._n_splits:
            raise PlaycatchDomainError(
                f"Amostras insuficientes ({len(features)}) para {self._n_splits} folds"
            )
        kfold = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._random_state)
        return kfold.split(features)


class TrainTestSplitStrategy(IValidationStrategy):
    """Validação simples via divisão treino/teste (holdout)."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
            self._test_size = test_size
        self._random_state = random_state

    def split(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            indices = np.arange(len(features))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=labels if labels.nunique() > 1 else None,
        )
        return iter([(train_idx, val_idx)])