
import pytest

from playcatch.chatbot.engine import ChatbotEngine
from playcatch.chatbot.interfaces import ILLMClient
from playcatch.domain.entities import ChatMessage, ChatRole, EmotionLabel
from playcatch.domain.exceptions import ChatbotError


class FakeLLMClient(ILLMClient):

    def __init__(self, fixed_response: str) -> None:
        self._fixed_response = fixed_response

    def generate(self, prompt: str) -> str:
        return self._fixed_response


class TestChatbotEngine:

    def test_classify_emotion_parses_valid_json(self) -> None:
        fake_client = FakeLLMClient(fixed_response='{"emotion": "happy"}')
        engine = ChatbotEngine(llm_client=fake_client)

        emotion = engine.classify_emotion("Estou muito feliz hoje!")

        assert emotion == EmotionLabel.HAPPY

    def test_classify_emotion_handles_markdown_wrapped_json(self) -> None:
        fake_client = FakeLLMClient(fixed_response='```json\n{"emotion": "sad"}\n```')
        engine = ChatbotEngine(llm_client=fake_client)

        emotion = engine.classify_emotion("Estou muito triste")

        assert emotion == EmotionLabel.SAD

    def test_classify_emotion_raises_on_invalid_json(self) -> None:
        fake_client = FakeLLMClient(fixed_response="não sou um json")
        engine = ChatbotEngine(llm_client=fake_client)

        with pytest.raises(ChatbotError):
            engine.classify_emotion("qualquer mensagem")

    def test_classify_emotion_raises_on_unknown_emotion_value(self) -> None:
        fake_client = FakeLLMClient(fixed_response='{"emotion": "inexistente"}')
        engine = ChatbotEngine(llm_client=fake_client)

        with pytest.raises(ChatbotError):
            engine.classify_emotion("qualquer mensagem")

    def test_generate_reply_returns_llm_text(self) -> None:
        fake_client = FakeLLMClient(fixed_response="Que bom que você está feliz! 🎵")
        engine = ChatbotEngine(llm_client=fake_client)
        history = [ChatMessage(role=ChatRole.USER, content="Estou feliz hoje!")]

        reply = engine.generate_reply(history, EmotionLabel.HAPPY)

        assert reply == "Que bom que você está feliz! 🎵"

    def test_generate_reply_raises_on_empty_history(self) -> None:
        fake_client = FakeLLMClient(fixed_response="qualquer coisa")
        engine = ChatbotEngine(llm_client=fake_client)

        with pytest.raises(ChatbotError):
            engine.generate_reply([], EmotionLabel.HAPPY)