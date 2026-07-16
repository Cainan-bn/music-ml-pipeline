
from dataclasses import dataclass, field
from enum import Enum


class EmotionLabel(str, Enum):

    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CALM = "calm"
    ENERGETIC = "energetic"
    ROMANTIC = "romantic"


@dataclass(frozen=True)
class Lyrics:

    song_id: str
    text: str
    language: str = "pt"

    def __post_init__(self) -> None:
            if not self.text or not self.text.strip():
            raise ValueError(f"Lyrics vazia para song_id={self.song_id}")


@dataclass(frozen=True)
class SentimentResult:

    song_id: str
    emotion: EmotionLabel
    confidence: float
    emotion_distribution: dict[EmotionLabel, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
            if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Confidence deve estar entre 0 e 1, recebido: {self.confidence}"
            )


@dataclass(frozen=True)
class Song:

    song_id: str
    title: str
    artist: str
    lyrics: Lyrics | None = None
    sentiment: SentimentResult | None = None


@dataclass(frozen=True)
class UserMoodQuery:

    user_id: str
    target_emotion: EmotionLabel
    top_k: int = 10

    def __post_init__(self) -> None:
            if self.top_k <= 0:
            raise ValueError("top_k deve ser um inteiro positivo")


@dataclass(frozen=True)
class RecommendationResult:

    user_id: str
    songs: list[Song]
    strategy_used: str

class ChatRole(str, Enum):

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
            if not self.content or not self.content.strip():
            raise ValueError("Conteúdo da mensagem não pode ser vazio")


@dataclass(frozen=True)
class ChatbotReply:

    text: str
    detected_emotion: EmotionLabel
    recommendation: "RecommendationResult | None" = None