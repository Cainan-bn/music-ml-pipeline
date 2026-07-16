
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from playcatch.application.evaluation.model_evaluator import ModelEvaluator
from playcatch.application.strategies.feature_strategies import TfidfFeatureStrategy
from playcatch.application.strategies.validation_strategies import KFoldValidationStrategy
from playcatch.config.logging_config import configure_logging, get_logger
from playcatch.config.settings import settings
from playcatch.infrastructure.data_sources.csv_source import CsvDataSource
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    data_source = CsvDataSource(settings.data_source_path)
    raw_df = data_source.load()

    feature_extractor = TfidfFeatureStrategy(max_features=500)
    features = feature_extractor.fit_transform(raw_df["lyrics"].astype(str).tolist())
    labels = raw_df["emotion"]

    evaluator = ModelEvaluator(validation_strategy=KFoldValidationStrategy(n_splits=5))
    results = evaluator.evaluate(
        model_factory=lambda: LightGbmRecommender(num_boost_round=50),
        features=features,
        labels=labels,
    )

    print(f"\n📊 Acurácia média: {results['mean_accuracy']:.2%}\n")
    print("Acurácia por fold:", [f"{acc:.2%}" for acc in results["fold_accuracies"]])
    print("\n--- Relatório do último fold ---")
    print(results["classification_reports"][-1])


if __name__ == "__main__":
    main()