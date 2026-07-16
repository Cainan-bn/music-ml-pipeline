
import json
import re

from playcatch.config.logging_config import get_logger
from playcatch.chatbot.interfaces import IChatbotEngine, ILLMClient
from playcatch.domain.entities import ChatMessage, ChatRole, EmotionLabel
from playcatch.domain.exceptions import ChatbotError

logger = get_logger(__name__)

_VALID_EMOTIONS = ", ".join(e.value for e in EmotionLabel)

_EMOTION_PROMPT_TEMPLATE = """Você é um classificador de emoções para um chatbot de música.
Analise a mensagem do usuário abaixo e identifique a emoção predominante.

Responda ESTRITAMENTE em JSON, sem nenhum texto adicional, no formato:
{{"emotion": "<uma das opções: {valid_emotions}>"}}

Mensagem do usuário: "{message}"
"""

_REPLY_PROMPT_TEMPLATE = """Você é o assistente musical da Playcatch, uma plataforma de streaming.
Converse de forma calorosa e natural sobre música com o usuário.
A emoção detectada na última mensagem dele foi: {emotion}.

Histórico da conversa:
{history}

Responda à última mensagem do usuário de forma breve (2-4 frases), amigável,
e conecte a resposta ao humor detectado, sem inventar nomes de músicas.
"""


class ChatbotEngine(IChatbotEngine):

    def __init__(self, llm_client: ILLMClient) -> None:
            self._llm_client = llm_client

    def classify_emotion(self, message: str) -> EmotionLabel:
            prompt = _EMOTION_PROMPT_TEMPLATE.format(
            valid_emotions=_VALID_EMOTIONS, message=message
        )

        try:
            raw_response = self._llm_client.generate(prompt)
        except ChatbotError:
            logger.exception("Falha ao classificar emoção via LLM")
            raise

        emotion_value = self._extract_emotion_json(raw_response)

        try:
            return EmotionLabel(emotion_value)
        except ValueError as exc:
            raise ChatbotError(
                f"LLM retornou emoção inválida: '{emotion_value}'"
            ) from exc

    def generate_reply(
        self, history: list[ChatMessage], detected_emotion: EmotionLabel
    ) -> str:
            if not history:
            raise ChatbotError("Histórico de conversa vazio; impossível gerar resposta")

        formatted_history = "\n".join(
            f"{msg.role.value}: {msg.content}" for msg in history
        )
        prompt = _REPLY_PROMPT_TEMPLATE.format(
            emotion=detected_emotion.value, history=formatted_history
        )

        try:
            return self._llm_client.generate(prompt)
        except ChatbotError:
            logger.exception("Falha ao gerar resposta via LLM")
            raise

    @staticmethod
    def _extract_emotion_json(raw_response: str) -> str:
            cleaned = re.sub(r"```json|```", "", raw_response).strip()
        try:
            parsed = json.loads(cleaned)
            return str(parsed["emotion"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ChatbotError(
                f"Resposta do LLM não é um JSON válido de emoção: '{raw_response}'"
            ) from exc