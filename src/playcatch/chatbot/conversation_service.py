
from playcatch.application.pipelines.inference_pipeline import InferencePipeline
from playcatch.chatbot.interfaces import IChatbotEngine
from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import (
    ChatMessage,
    ChatRole,
    ChatbotReply,
    EmotionLabel,
    UserMoodQuery,
)
from playcatch.domain.exceptions import ChatbotError, PlaycatchDomainError

logger = get_logger(__name__)

_FALLBACK_EMOTION = EmotionLabel.CALM


class ConversationService:

    def __init__(self, chatbot_engine: IChatbotEngine, inference_pipeline: InferencePipeline) -> None:
            self._chatbot_engine = chatbot_engine
        self._inference_pipeline = inference_pipeline

    def handle_user_message(
        self, user_id: str, history: list[ChatMessage], top_k: int = 5
    ) -> ChatbotReply:
            if not history or history[-1].role != ChatRole.USER:
            raise ChatbotError("A última mensagem do histórico deve ser do usuário")

        last_message = history[-1].content

        detected_emotion = self._classify_with_fallback(last_message)

        recommendation = self._try_get_recommendation(user_id, detected_emotion, top_k)

        reply_text = self._chatbot_engine.generate_reply(history, detected_emotion)

        return ChatbotReply(
            text=reply_text,
            detected_emotion=detected_emotion,
            recommendation=recommendation,
        )

    def _classify_with_fallback(self, message: str) -> EmotionLabel:
            try:
            return self._chatbot_engine.classify_emotion(message)
        except ChatbotError as exc:
            logger.warning(
                "Falha ao classificar emoção, usando fallback '%s': %s",
                _FALLBACK_EMOTION.value,
                exc,
            )
            return _FALLBACK_EMOTION

    def _try_get_recommendation(
        self, user_id: str, emotion: EmotionLabel, top_k: int
    ):
            query = UserMoodQuery(user_id=user_id, target_emotion=emotion, top_k=top_k)
        try:
            return self._inference_pipeline.run(query)
        except PlaycatchDomainError as exc:
            logger.warning("Não foi possível gerar recomendação para o chat: %s", exc)
            return None