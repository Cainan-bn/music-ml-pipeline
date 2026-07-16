
from typing import Callable

from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import EmotionLabel, Lyrics, SentimentResult
from playcatch.domain.exceptions import SentimentAnalysisError
from playcatch.domain.interfaces import ISentimentAnalyzer

logger = get_logger(__name__)

ModelRunner = Callable[[str], dict[str, float]]


class HuggingFaceSentimentAnalyzer(ISentimentAnalyzer):

    def __init__(self, model_runner: ModelRunner) -> None:
            self._model_runner = model_runner

    def analyze(self, lyrics: Lyrics) -> SentimentResult:
            try:
            distribution_raw = self._model_runner(lyrics.text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha no modelo de sentimento para song_id=%s: %s", lyrics.song_id, exc)
            raise SentimentAnalysisError(
                f"Falha ao analisar sentimento da música '{lyrics.song_id}'"
            ) from exc

        if not distribution_raw:
            raise SentimentAnalysisError(
                f"Modelo retornou distribuição vazia para song_id={lyrics.song_id}"
            )

        distribution = {
            EmotionLabel(label): score for label, score in distribution_raw.items()
        }
        predominant_emotion = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        confidence = distribution[predominant_emotion]

        return SentimentResult(
            song_id=lyrics.song_id,
            emotion=predominant_emotion,
            confidence=confidence,
            emotion_distribution=distribution,
        )

    def analyze_batch(self, lyrics_list: list[Lyrics]) -> list[SentimentResult]:
            results: list[SentimentResult] = []
        for lyrics in lyrics_list:
            try:
                results.append(self.analyze(lyrics))
            except SentimentAnalysisError as exc:
                logger.warning("Ignorando música com erro de análise: %s", exc)
        return results