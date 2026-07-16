
from playcatch.config.logging_config import get_logger
from playcatch.domain.entities import EmotionLabel, Lyrics, SentimentResult
from playcatch.domain.exceptions import SentimentAnalysisError
from playcatch.domain.interfaces import ISentimentAnalyzer

logger = get_logger(__name__)


class GroundTruthSentimentAnalyzer(ISentimentAnalyzer):

    def __init__(self, emotion_map: dict[str, str]) -> None:
        self._emotion_map = emotion_map

    def analyze(self, lyrics: Lyrics) -> SentimentResult:
            raw_emotion = self._emotion_map.get(lyrics.song_id)
        if raw_emotion is None:
            raise SentimentAnalysisError(
                f"Nenhum rótulo de emoção conhecido para song_id={lyrics.song_id}"
            )
        try:
            emotion = EmotionLabel(raw_emotion.strip().lower())
        except ValueError as exc:
            raise SentimentAnalysisError(
                f"Emoção '{raw_emotion}' inválida para song_id={lyrics.song_id}"
            ) from exc

        return SentimentResult(
            song_id=lyrics.song_id,
            emotion=emotion,
            confidence=1.0,
            emotion_distribution={emotion: 1.0},
        )

    def analyze_batch(self, lyrics_list: list[Lyrics]) -> list[SentimentResult]:
            results: list[SentimentResult] = []
        for lyrics in lyrics_list:
            try:
                results.append(self.analyze(lyrics))
            except SentimentAnalysisError as exc:
                logger.warning("Ignorando música sem rótulo: %s", exc)
        return results