
import pandas as pd

from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import RecommendationResult, UserMoodQuery
from playcatch.domain.exceptions import PlaycatchDomainError
from playcatch.domain.interfaces import IDataSource, IFeatureExtractor, IRecommenderModel

logger = get_logger(__name__)

METADATA_COLUMNS = ("song_id", "title", "artist")


class InferencePipeline:

    def __init__(
        self,
        candidates_source: IDataSource,
        feature_extractor: IFeatureExtractor,
        recommender_model: IRecommenderModel,
    ) -> None:
            self._candidates_source = candidates_source
        self._feature_extractor = feature_extractor
        self._recommender_model = recommender_model

    def run(self, query: UserMoodQuery) -> RecommendationResult:
            logger.info(
            "Iniciando pipeline de inferência para user_id=%s, emoção=%s",
            query.user_id,
            query.target_emotion.value,
        )

        try:
            raw_candidates = self._candidates_source.load()
        except PlaycatchDomainError:
            logger.exception("Falha ao carregar candidatos na inferência")
            raise

        try:
            feature_matrix = self._feature_extractor.transform(
                raw_candidates["lyrics"].astype(str).tolist()
            )
        except PlaycatchDomainError:
            logger.exception("Falha na extração de features na inferência")
            raise

        candidates_with_features = self._merge_metadata_and_features(
            raw_candidates, feature_matrix
        )

        try:
            result = self._recommender_model.predict(query, candidates_with_features)
        except PlaycatchDomainError:
            logger.exception("Falha na geração de recomendações")
            raise

        logger.info("Pipeline de inferência concluído: %d recomendações", len(result.songs))
        return result

    @staticmethod
    def _merge_metadata_and_features(
        raw_candidates: pd.DataFrame, feature_matrix: pd.DataFrame
    ) -> pd.DataFrame:
            metadata = raw_candidates[list(METADATA_COLUMNS)].reset_index(drop=True)
        features = feature_matrix.reset_index(drop=True)
        return pd.concat([metadata, features], axis=1)