
import pandas as pd
import pytest

from playcatch.application.pipelines.inference_pipeline import InferencePipeline
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.chatbot.conversation_service import ConversationService
from playcatch.chatbot.engine import ChatbotEngine
from playcatch.chatbot.interfaces import ILLMClient
from playcatch.domain.entities import ChatMessage, ChatRole
from playcatch.domain.interfaces import IDataSource
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender


class InMemoryCandidatesSource(IDataSource):

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def load(self) -> pd.DataFrame:
        return self._df


class ScriptedLLMClient(ILLMClient):

    def __init__(self, scripted_responses: list[str]) -> None:
        self._responses = iter(scripted_responses)

    def generate(self, prompt: str) -> str:
        return next(self._responses)


@pytest.fixture
def songs_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": [f"s{i}" for i in range(6)],
            "title": [f"Música {i}" for i in range(6)],
            "artist": [f"Artista {i}" for i in range(6)],
            "lyrics": [
                "Estou muito feliz, que alegria, dia de sol",
                "Que tristeza profunda, saudade e dor",
                "Estou muito feliz, sorriso no rosto, alegria",
                "Saudade e dor, tristeza sem fim, chorar",
                "Feliz e cheio de alegria, sol brilhando",
                "Tristeza, dor e saudade, chorar sempre",
            ],
        }
    )


class TestConversationServiceIntegration:

    def test_full_conversation_flow_returns_reply_with_recommendation(
        self, songs_dataset: pd.DataFrame
    ) -> None:
        feature_extractor = TfidfFeatureStrategy(max_features=50)
        features = feature_extractor.fit_transform(songs_dataset["lyrics"].tolist())
        labels = pd.Series(["happy", "sad", "happy", "sad", "happy", "sad"])

        recommender_model = LightGbmRecommender(num_boost_round=20)
        recommender_model.train(features, labels)

        inference_pipeline = InferencePipeline(
            candidates_source=InMemoryCandidatesSource(songs_dataset),
            feature_extractor=feature_extractor,
            recommender_model=recommender_model,
        )

        scripted_llm = ScriptedLLMClient(
            scripted_responses=[
                '{"emotion": "happy"}',
                "Que alegria saber que você está feliz! Separei umas músicas animadas.",
            ]
        )
        chatbot_engine = ChatbotEngine(llm_client=scripted_llm)
        conversation_service = ConversationService(
            chatbot_engine=chatbot_engine, inference_pipeline=inference_pipeline
        )

        history = [ChatMessage(role=ChatRole.USER, content="Estou muito feliz hoje!")]
        reply = conversation_service.handle_user_message(user_id="u1", history=history, top_k=3)

        assert reply.detected_emotion.value == "happy"
        assert "feliz" in reply.text.lower() or "alegria" in reply.text.lower()
        assert reply.recommendation is not None
        assert len(reply.recommendation.songs) == 3

    def test_conversation_falls_back_gracefully_when_emotion_classification_fails(
        self, songs_dataset: pd.DataFrame
    ) -> None:
        feature_extractor = TfidfFeatureStrategy(max_features=50)
        features = feature_extractor.fit_transform(songs_dataset["lyrics"].tolist())
        labels = pd.Series(["happy", "sad", "happy", "sad", "happy", "sad"])

        recommender_model = LightGbmRecommender(num_boost_round=20)
        recommender_model.train(features, labels)

        inference_pipeline = InferencePipeline(
            candidates_source=InMemoryCandidatesSource(songs_dataset),
            feature_extractor=feature_extractor,
            recommender_model=recommender_model,
        )

        # Primeira resposta (classificação) é inválida de propósito;
        # segunda resposta (geração de texto) é válida.
        scripted_llm = ScriptedLLMClient(
            scripted_responses=["texto quebrado, não é json", "Vamos conversar sobre música!"]
        )
        chatbot_engine = ChatbotEngine(llm_client=scripted_llm)
        conversation_service = ConversationService(
            chatbot_engine=chatbot_engine, inference_pipeline=inference_pipeline
        )

        history = [ChatMessage(role=ChatRole.USER, content="mensagem qualquer")]
        reply = conversation_service.handle_user_message(user_id="u2", history=history)

        assert reply.detected_emotion.value == "calm"  # fallback
        assert reply.text == "Vamos conversar sobre música!"