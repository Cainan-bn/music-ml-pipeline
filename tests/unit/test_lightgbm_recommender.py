
import pandas as pd
import pytest
import joblib

from playcatch.domain.entities import EmotionLabel, UserMoodQuery
from playcatch.domain.exceptions import InferenceError
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender


@pytest.fixture
def trained_recommender() -> LightGbmRecommender:
    features = pd.DataFrame(
        {
            "feat_a": [1.0, 0.0, 1.0, 0.0, 0.9, 0.1] * 3,
            "feat_b": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9] * 3,
        }
    )
    labels = pd.Series(["happy", "sad", "happy", "sad", "happy", "sad"] * 3)

    model = LightGbmRecommender(num_boost_round=20)
    model.train(features, labels)
    return model


class TestLightGbmRecommender:

    def test_predict_returns_top_k_songs(self, trained_recommender: LightGbmRecommender) -> None:
        candidates = pd.DataFrame(
            {
                "song_id": ["s1", "s2", "s3"],
                "title": ["Música 1", "Música 2", "Música 3"],
                "artist": ["A", "B", "C"],
                "feat_a": [1.0, 0.0, 0.8],
                "feat_b": [0.0, 1.0, 0.2],
            }
        )
        query = UserMoodQuery(user_id="u1", target_emotion=EmotionLabel.HAPPY, top_k=2)

        result = trained_recommender.predict(query, candidates)

        assert result.user_id == "u1"
        assert len(result.songs) == 2
        assert result.strategy_used == "lightgbm_ranking"

    def test_predict_raises_when_model_not_trained(self) -> None:
        model = LightGbmRecommender()
        query = UserMoodQuery(user_id="u1", target_emotion=EmotionLabel.HAPPY, top_k=1)
        candidates = pd.DataFrame(
            {"song_id": ["s1"], "title": ["T"], "artist": ["A"], "feat_a": [1.0]}
        )

        with pytest.raises(InferenceError):
            model.predict(query, candidates)

    def test_predict_raises_on_unknown_emotion(
        self, trained_recommender: LightGbmRecommender
    ) -> None:
        candidates = pd.DataFrame(
            {
                "song_id": ["s1"],
                "title": ["T"],
                "artist": ["A"],
                "feat_a": [1.0],
                "feat_b": [0.0],
            }
        )
        query = UserMoodQuery(user_id="u1", target_emotion=EmotionLabel.ANGRY, top_k=1)

        with pytest.raises(InferenceError):
            trained_recommender.predict(query, candidates)