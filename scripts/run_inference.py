
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playcatch.application.pipelines.inference_pipeline import InferencePipeline
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.config.logging_config import configure_logging, get_logger
from playcatch.config.settings import settings
from playcatch.domain.entities import EmotionLabel, UserMoodQuery
from playcatch.infrastructure.data_sources.csv_source import CsvDataSource
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender

configure_logging()
logger = get_logger(__name__)

ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    candidates_source = CsvDataSource(settings.data_source_path)
    feature_extractor = TfidfFeatureStrategy.load(str(ARTIFACTS_DIR / "vectorizer.joblib"))
    recommender_model = LightGbmRecommender.load(str(ARTIFACTS_DIR / "recommender.joblib"))

    pipeline = InferencePipeline(
        candidates_source=candidates_source,
        feature_extractor=feature_extractor,
        recommender_model=recommender_model,
    )

    query = UserMoodQuery(user_id="demo_user", target_emotion=EmotionLabel.HAPPY, top_k=5)
    result = pipeline.run(query)

    logger.info("Recomendações para emoção '%s':", query.target_emotion.value)
    for song in result.songs:
        logger.info(" - %s (%s)", song.title, song.artist)


if __name__ == "__main__":
    main()