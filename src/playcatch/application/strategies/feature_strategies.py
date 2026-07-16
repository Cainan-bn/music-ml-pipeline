import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from playcatch.config.logging_config import get_logger
from playcatch.domain.exceptions import FeatureExtractionError
from playcatch.domain.interfaces import IFeatureExtractor

logger = get_logger(__name__)


class TfidfFeatureStrategy(IFeatureExtractor):
    """Extrai features via TF-IDF sobre as letras das músicas."""

    def __init__(self, max_features: int = 500) -> None:
            self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._is_fitted = False

    def fit_transform(self, texts: list[str]) -> pd.DataFrame:
            if not texts:
            raise FeatureExtractionError("Lista de textos vazia para extração de features")
        matrix = self._vectorizer.fit_transform(texts)
        self._is_fitted = True
        return pd.DataFrame(
            matrix.toarray(), columns=self._vectorizer.get_feature_names_out()
        )

    def transform(self, texts: list[str]) -> pd.DataFrame:
            if not self._is_fitted:
            raise FeatureExtractionError(
                "TfidfFeatureStrategy não foi ajustada. Chame fit_transform primeiro."
            )
        matrix = self._vectorizer.transform(texts)
        return pd.DataFrame(
            matrix.toarray(), columns=self._vectorizer.get_feature_names_out()
        )

    def save(self, path: str) -> None:
            if not self._is_fitted:
            raise FeatureExtractionError("Não é possível salvar um vetorizador não ajustado")
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeatureStrategy":
            instance = cls()
        instance._vectorizer = joblib.load(path)
        instance._is_fitted = True
        return instance

    def save(self, path: str) -> None:
        """Salva o vetorizador ajustado em disco.

        Args:
            path: Caminho do arquivo de destino (ex: 'artifacts/vectorizer.joblib').

        Raises:
            FeatureExtractionError: Se o vetorizador ainda não tiver sido ajustado.
        """
        if not self._is_fitted:
            raise FeatureExtractionError("Não é possível salvar um vetorizador não ajustado")
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "TfidfFeatureStrategy":
        """Carrega um vetorizador previamente salvo.

        Args:
            path: Caminho do arquivo salvo.

        Returns:
            Instância de TfidfFeatureStrategy pronta para uso (transform).
        """
        instance = cls()
        instance._vectorizer = joblib.load(path)
        instance._is_fitted = True
        return instance