
import pandas as pd
import pytest

from playcatch.application.pipelines.training_pipeline import TrainingPipeline
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.domain.exceptions import PlaycatchDomainError
from playcatch.domain.interfaces import IDataSource, IRecommenderModel
from playcatch.infrastructure.sentiment.huggingface_analyzer import (
    HuggingFaceSentimentAnalyzer,
)


class InMemoryDataSource(IDataSource):

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load(self) -> pd.DataFrame:
        return self._df


class DummyRecommenderModel(IRecommenderModel):

    def __init__(self) -> None:
        self.was_trained = False
        self.received_features: pd.DataFrame | None = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self.was_trained = True
        self.received_features = features

    def predict(self, query, candidates):  # noqa: D102 - não exercido neste teste
        raise NotImplementedError


def alternating_sentiment_runner(text: str) -> dict[str, float]:
    """Runner determinístico: classifica como 'happy' ou 'sad' pela palavra-chave."""
    if "feliz" in text.lower():
        return {"happy": 0.8, "sad": 0.2}
    return {"happy": 0.2, "sad": 0.8}


class TestTrainingPipelineIntegration:

    def test_full_pipeline_trains_model_successfully(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        data_source = InMemoryDataSource(sample_dataframe)
        sentiment_analyzer = HuggingFaceSentimentAnalyzer(
            model_runner=alternating_sentiment_runner
        )
        feature_extractor = TfidfFeatureStrategy(max_features=50)
        recommender_model = DummyRecommenderModel()

        pipeline = TrainingPipeline(
            data_source=data_source,
            sentiment_analyzer=sentiment_analyzer,
            feature_extractor=feature_extractor,
            recommender_model=recommender_model,
        )

        pipeline.run()

        assert recommender_model.was_trained is True
        assert recommender_model.received_features is not None
        assert len(recommender_model.received_features) == 2

    def test_pipeline_raises_on_empty_dataset(self) -> None:
        empty_df = pd.DataFrame(columns=["song_id", "title", "artist", "lyrics"])
        data_source = InMemoryDataSource(empty_df)
        sentiment_analyzer = HuggingFaceSentimentAnalyzer(
            model_runner=alternating_sentiment_runner
        )
        pipeline = TrainingPipeline(
            data_source=data_source,
            sentiment_analyzer=sentiment_analyzer,
            feature_extractor=TfidfFeatureStrategy(),
            recommender_model=DummyRecommenderModel(),
        )

        with pytest.raises(PlaycatchDomainError):
            pipeline.run()