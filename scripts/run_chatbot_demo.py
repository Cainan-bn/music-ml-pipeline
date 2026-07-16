
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playcatch.application.pipelines.inference_pipeline import InferencePipeline
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.chatbot.conversation_service import ConversationService
from playcatch.chatbot.engine import ChatbotEngine
from playcatch.config.logging_config import configure_logging, get_logger
from playcatch.config.settings import settings
from playcatch.domain.entities import ChatMessage, ChatRole
from playcatch.infrastructure.data_sources.csv_source import CsvDataSource
from playcatch.infrastructure.llm.gemini_client import GeminiLLMClient
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender

configure_logging()
logger = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    candidates_source = CsvDataSource(settings.data_source_path)
    feature_extractor = TfidfFeatureStrategy.load(str(ARTIFACTS_DIR / "vectorizer.joblib"))
    recommender_model = LightGbmRecommender.load(str(ARTIFACTS_DIR / "recommender.joblib"))

    inference_pipeline = InferencePipeline(
        candidates_source=candidates_source,
        feature_extractor=feature_extractor,
        recommender_model=recommender_model,
    )

    llm_client = GeminiLLMClient(
        api_key=settings.gemini_api_key, model_name=settings.gemini_model_name
    )
    chatbot_engine = ChatbotEngine(llm_client=llm_client)
    conversation_service = ConversationService(
        chatbot_engine=chatbot_engine, inference_pipeline=inference_pipeline
    )

    history: list[ChatMessage] = []
    print("🎵 Chatbot Playcatch — digite 'sair' para encerrar\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in {"sair", "exit", "quit"}:
            break

        history.append(ChatMessage(role=ChatRole.USER, content=user_input))
        reply = conversation_service.handle_user_message(user_id="cli_user", history=history)
        history.append(ChatMessage(role=ChatRole.ASSISTANT, content=reply.text))

        print(f"\nPlaycatch Bot ({reply.detected_emotion.value}): {reply.text}")
        if reply.recommendation:
            print("🎧 Recomendações:")
            for song in reply.recommendation.songs:
                print(f"   - {song.title} ({song.artist})")
        print()


if __name__ == "__main__":
    main()