
import pandas as pd

from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import Lyrics
from playcatch.domain.exceptions import PlaycatchDomainError
from playcatch.domain.interfaces import (
    IDataSource,
    IFeatureExtractor,
    IRecommenderModel,
    ISentimentAnalyzer,
)

logger = get_logger(__name__)


class TrainingPipeline:

    def __init__(
        self,
        data_source: IDataSource,
        sentiment_analyzer: ISentimentAnalyzer,
        feature_extractor: IFeatureExtractor,
        recommender_model: IRecommenderModel,
    ) -> None:
            self._data_source = data_source
        self._sentiment_analyzer = sentiment_analyzer
        self._feature_extractor = feature_extractor
        self._recommender_model = recommender_model

    def run(self) -> None:
            logger.info("Iniciando pipeline de treino")

        try:
            raw_df = self._data_source.load()
        except PlaycatchDomainError:
            logger.exception("Falha na etapa de ingestão de dados")
            raise

        lyrics_list = self._build_lyrics_entities(raw_df)

        try:
            sentiment_results = self._sentiment_analyzer.analyze_batch(lyrics_list)
        except PlaycatchDomainError:
            logger.exception("Falha na etapa de análise de sentimento")
            raise

        if not sentiment_results:
            raise PlaycatchDomainError(
                "Nenhum resultado de sentimento gerado; abortando treino"
            )

        texts = [lyrics.text for lyrics in lyrics_list]
        labels = pd.Series([r.emotion.value for r in sentiment_results])

        try:
            features = self._feature_extractor.fit_transform(texts)
        except PlaycatchDomainError:
            logger.exception("Falha na etapa de extração de features")
            raise

        try:
            self._recommender_model.train(features=features, labels=labels)
        except PlaycatchDomainError:
            logger.exception("Falha na etapa de treino do modelo")
            raise

        logger.info("Pipeline de treino concluído com sucesso")

    @staticmethod
    def _build_lyrics_entities(raw_df: pd.DataFrame) -> list[Lyrics]:
            lyrics_list: list[Lyrics] = []
        for _, row in raw_df.iterrows():
            try:
                lyrics_list.append(
                    Lyrics(song_id=str(row["song_id"]), text=str(row["lyrics"]))
                )
            except ValueError as exc:
                logger.warning("Registro descartado por letra inválida: %s", exc)
        return lyrics_list