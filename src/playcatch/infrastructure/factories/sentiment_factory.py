
from playcatch.domain.exceptions import ModelNotFoundError
from playcatch.domain.interfaces import ISentimentAnalyzer
from playcatch.infrastructure.sentiment.ground_truth_analyzer import (
    GroundTruthSentimentAnalyzer,
)
from playcatch.infrastructure.sentiment.huggingface_analyzer import (
    HuggingFaceSentimentAnalyzer,
)


class SentimentAnalyzerFactory:

    _registry: dict[str, type[ISentimentAnalyzer]] = {
        "huggingface": HuggingFaceSentimentAnalyzer,
        "ground_truth": GroundTruthSentimentAnalyzer,
    }

    @classmethod
    def create(cls, analyzer_type: str, **kwargs: object) -> ISentimentAnalyzer:
            analyzer_cls = cls._registry.get(analyzer_type)
        if analyzer_cls is None:
            raise ModelNotFoundError(
                f"Analisador de sentimento '{analyzer_type}' não registrado. "
                f"Opções disponíveis: {list(cls._registry.keys())}"
            )
        return analyzer_cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def register(cls, name: str, analyzer_cls: type[ISentimentAnalyzer]) -> None:
            cls._registry[name] = analyzer_cls