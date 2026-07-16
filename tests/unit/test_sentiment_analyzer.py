
import pytest

from playcatch.domain.entities import EmotionLabel, Lyrics
from playcatch.domain.exceptions import SentimentAnalysisError
from playcatch.infrastructure.sentiment.huggingface_analyzer import (
    HuggingFaceSentimentAnalyzer,
)


def fake_model_runner_happy(text: str) -> dict[str, float]:
    """Mock de modelo que sempre retorna alta confiança em 'happy'."""
    return {"happy": 0.9, "sad": 0.05, "angry": 0.05}


def fake_model_runner_empty(text: str) -> dict[str, float]:
    return {}


class TestHuggingFaceSentimentAnalyzer:

    def test_analyze_returns_predominant_emotion(self, sample_lyrics: Lyrics) -> None:
        analyzer = HuggingFaceSentimentAnalyzer(model_runner=fake_model_runner_happy)

        result = analyzer.analyze(sample_lyrics)

        assert result.emotion == EmotionLabel.HAPPY
        assert result.confidence == pytest.approx(0.9)
        assert result.song_id == sample_lyrics.song_id

    def test_analyze_raises_on_empty_distribution(self, sample_lyrics: Lyrics) -> None:
        analyzer = HuggingFaceSentimentAnalyzer(model_runner=fake_model_runner_empty)

        with pytest.raises(SentimentAnalysisError):
            analyzer.analyze(sample_lyrics)

    def test_analyze_batch_skips_invalid_entries(self, sample_lyrics: Lyrics) -> None:
        analyzer_ok = HuggingFaceSentimentAnalyzer(model_runner=fake_model_runner_happy)
        analyzer_fail = HuggingFaceSentimentAnalyzer(model_runner=fake_model_runner_empty)

        # Combina um analyzer que falha para provar que o batch não quebra tudo
        results_ok = analyzer_ok.analyze_batch([sample_lyrics])
        results_fail = analyzer_fail.analyze_batch([sample_lyrics])

        assert len(results_ok) == 1
        assert len(results_fail) == 0