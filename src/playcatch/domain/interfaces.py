from typing import Protocol
from abc import ABC, abstractmethod

import pandas as pd

from playcatch.domain.entities import (
    Lyrics,
    RecommendationResult,
    SentimentResult,
    UserMoodQuery,
)


class IDataSource(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Carrega o dataset bruto.

        raise NotImplementedError


class ISentimentAnalyzer(ABC):

    @abstractmethod
    def analyze(self, lyrics: Lyrics) -> SentimentResult:
            raise NotImplementedError

    @abstractmethod
    def analyze_batch(self, lyrics_list: list[Lyrics]) -> list[SentimentResult]:
            raise NotImplementedError


class IFeatureExtractor(ABC):

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> pd.DataFrame:
            raise NotImplementedError

    @abstractmethod
    def transform(self, texts: list[str]) -> pd.DataFrame:
            raise NotImplementedError


class IRecommenderModel(ABC):

    @abstractmethod
    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
            raise NotImplementedError

    @abstractmethod
    def predict(self, query: UserMoodQuery, candidates: pd.DataFrame) -> RecommendationResult:
            raise NotImplementedError


class IValidationStrategy(ABC):

    @abstractmethod
    def split(self, features: pd.DataFrame, labels: pd.Series):
            raise NotImplementedError

class IProbabilisticClassifier(Protocol):


    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
