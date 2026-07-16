
import pytest

from playcatch.domain.entities import EmotionLabel, Lyrics
from playcatch.domain.exceptions import SentimentAnalysisError
from playcatch.infrastructure.sentiment.ground_truth_analyzer import (
    GroundTruthSentimentAnalyzer,
)


class TestGroundTruthSentimentAnalyzer:

    def test_analyze_returns_known_emotion_with_full_confidence(self) -> None:
        analyzer = GroundTruthSentimentAnalyzer(emotion_map={"s1": "happy"})
        lyrics = Lyrics(song_id="s1", text="qualquer texto")

        result = analyzer.analyze(lyrics)

        assert result.emotion == EmotionLabel.HAPPY
        assert result.confidence == 1.0

    def test_analyze_raises_when_song_id_unknown(self) -> None:
        analyzer = GroundTruthSentimentAnalyzer(emotion_map={})
        lyrics = Lyrics(song_id="unknown", text="qualquer texto")

        with pytest.raises(SentimentAnalysisError):
            analyzer.analyze(lyrics)

    def test_analyze_batch_skips_unknown_entries(self) -> None:
        analyzer = GroundTruthSentimentAnalyzer(emotion_map={"s1": "sad"})
        lyrics_list = [
            Lyrics(song_id="s1", text="texto a"),
            Lyrics(song_id="s2", text="texto b"),
        ]

        results = analyzer.analyze_batch(lyrics_list)

        assert len(results) == 1
        assert results[0].emotion == EmotionLabel.SAD