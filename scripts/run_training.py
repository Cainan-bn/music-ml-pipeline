
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playcatch.application.pipelines.training_pipeline import TrainingPipeline
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.config.logging_config import configure_logging, get_logger
from playcatch.config.settings import settings
from playcatch.infrastructure.data_sources.csv_source import CsvDataSource
from playcatch.infrastructure.factories.model_factory import ModelFactory
from playcatch.infrastructure.factories.sentiment_factory import SentimentAnalyzerFactory

configure_logging()
logger = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    data_source = CsvDataSource(settings.data_source_path)

    raw_df = data_source.load()
    emotion_map = dict(zip(raw_df["song_id"].astype(str), raw_df["emotion"]))

    sentiment_analyzer = SentimentAnalyzerFactory.create(
        "ground_truth", emotion_map=emotion_map
    )
    feature_extractor = TfidfFeatureStrategy(max_features=500)
    recommender_model = ModelFactory.create(settings.recommender_model_type)

    pipeline = TrainingPipeline(
        data_source=data_source,
        sentiment_analyzer=sentiment_analyzer,
        feature_extractor=feature_extractor,
        recommender_model=recommender_model,
    )
    pipeline.run()

    feature_extractor.save(str(ARTIFACTS_DIR / "vectorizer.joblib"))
    recommender_model.save(str(ARTIFACTS_DIR / "recommender.joblib"))
    logger.info("Artefatos salvos em '%s'", ARTIFACTS_DIR.resolve())


if __name__ == "__main__":
    main()