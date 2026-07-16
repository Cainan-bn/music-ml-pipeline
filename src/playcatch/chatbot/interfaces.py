
from abc import ABC, abstractmethod

from playcatch.domain.entities import ChatMessage, ChatbotReply, EmotionLabel


class ILLMClient(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
            raise NotImplementedError


class IChatbotEngine(ABC):

    @abstractmethod
    def classify_emotion(self, message: str) -> EmotionLabel:
            raise NotImplementedError

    @abstractmethod
    def generate_reply(
        self, history: list[ChatMessage], detected_emotion: EmotionLabel
    ) -> str:
            raise NotImplementedError