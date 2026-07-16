import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import RecommendationResult, Song, UserMoodQuery
from playcatch.domain.exceptions import InferenceError, SchemaValidationError
from playcatch.domain.interfaces import IRecommenderModel

logger = get_logger(__name__)

METADATA_COLUMNS = ("song_id", "title", "artist")


class LightGbmRecommender(IRecommenderModel):

    def __init__(self, num_boost_round: int = 100, params: dict | None = None) -> None:
            self._num_boost_round = num_boost_round
        self._params = params or {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "verbosity": -1,
            "learning_rate": 0.1,
        }
        self._model: lgb.Booster | None = None
        self._label_encoder = LabelEncoder()
        self._feature_columns: list[str] = []

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
            if features.empty or labels.empty:
            raise InferenceError("Features ou labels vazios; impossível treinar")
        if len(features) != len(labels):
            raise InferenceError(
                f"Tamanho incompatível: features={len(features)}, labels={len(labels)}"
            )

        try:
            encoded_labels = self._label_encoder.fit_transform(labels)
            self._feature_columns = list(features.columns)

            train_dataset = lgb.Dataset(features, label=encoded_labels)
            params = {
                **self._params,
                "num_class": len(self._label_encoder.classes_),
            }
            self._model = lgb.train(
                params=params,
                train_set=train_dataset,
                num_boost_round=self._num_boost_round,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha ao treinar LightGbmRecommender: %s", exc)
            raise InferenceError(f"Falha no treino do modelo LightGBM: {exc}") from exc

        logger.info(
            "LightGbmRecommender treinado com %d amostras e %d classes",
            len(features),
            len(self._label_encoder.classes_),
        )

    def predict(self, query: UserMoodQuery, candidates: pd.DataFrame) -> RecommendationResult:
        if self._model is None:
            raise InferenceError("Modelo não treinado. Chame train() antes de predict().")

        missing_meta = set(METADATA_COLUMNS) - set(candidates.columns)
        if missing_meta:
            raise SchemaValidationError(
                f"Colunas de metadados ausentes nos candidatos: {missing_meta}"
            )

        target_emotion = query.target_emotion.value
        if target_emotion not in self._label_encoder.classes_:
            raise InferenceError(
                f"Emoção-alvo '{target_emotion}' desconhecida pelo modelo. "
                f"Emoções conhecidas: {list(self._label_encoder.classes_)}"
            )

        try:
            feature_matrix = self._align_features(candidates)
            probabilities = self._model.predict(feature_matrix)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha na predição do LightGbmRecommender: %s", exc)
            raise InferenceError(f"Falha ao gerar predições: {exc}") from exc

        target_class_idx = list(self._label_encoder.classes_).index(target_emotion)
        target_scores = probabilities[:, target_class_idx]

        ranked_df = candidates.copy()
        ranked_df["_affinity_score"] = target_scores
        ranked_df = ranked_df.sort_values("_affinity_score", ascending=False).head(
            query.top_k
        )

        recommended_songs = [
            Song(
                song_id=str(row["song_id"]),
                title=str(row["title"]),
                artist=str(row["artist"]),
            )
            for _, row in ranked_df.iterrows()
        ]

        logger.info(
            "Geradas %d recomendações para user_id=%s (emoção-alvo=%s)",
            len(recommended_songs),
            query.user_id,
            target_emotion,
        )

        return RecommendationResult(
            user_id=query.user_id,
            songs=recommended_songs,
            strategy_used="lightgbm_ranking",
        )

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
            if self._model is None:
            raise InferenceError("Modelo não treinado. Chame train() antes de predict_proba().")

        aligned = features.reindex(columns=self._feature_columns, fill_value=0.0)
        probabilities = self._model.predict(aligned)
        return pd.DataFrame(
            probabilities, columns=self._label_encoder.classes_, index=features.index
        )
    
    def _align_features(self, candidates: pd.DataFrame) -> pd.DataFrame:
            feature_only = candidates.drop(columns=list(METADATA_COLUMNS), errors="ignore")
        aligned = feature_only.reindex(columns=self._feature_columns, fill_value=0.0)
        return aligned

    def save(self, path: str) -> None:
        if self._model is None:
            raise InferenceError("Não é possível salvar um modelo não treinado")
        joblib.dump(
            {
                "model": self._model,
                "label_encoder": self._label_encoder,
                "feature_columns": self._feature_columns,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LightGbmRecommender":
            payload = joblib.load(path)
        instance = cls()
        instance._model = payload["model"]
        instance._label_encoder = payload["label_encoder"]
        instance._feature_columns = payload["feature_columns"]
        return instance