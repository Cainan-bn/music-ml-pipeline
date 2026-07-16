
from pathlib import Path

import pandas as pd

from playcatch.config.logging_config import get_logger
from playcatch.domain.exceptions import DataSourceError
from playcatch.domain.interfaces import IDataSource

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"song_id", "title", "artist", "lyrics"}


class CsvDataSource(IDataSource):

    def __init__(self, file_path: str) -> None:
            self._file_path = Path(file_path)

    def load(self) -> pd.DataFrame:
            if not self._file_path.exists():
            logger.error("Arquivo de dados não encontrado: %s", self._file_path)
            raise DataSourceError(f"Arquivo não encontrado: {self._file_path}")

        try:
            df = pd.read_csv(self._file_path)
        except Exception as exc:  # noqa: BLE001 - convertido para exceção de domínio
            logger.error("Falha ao ler CSV: %s", exc)
            raise DataSourceError(f"Falha ao ler CSV '{self._file_path}': {exc}") from exc

        if df.empty:
            raise DataSourceError(f"Dataset vazio em '{self._file_path}'")

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataSourceError(
                f"Colunas obrigatórias ausentes no dataset: {missing}"
            )

        logger.info("Dataset carregado com sucesso: %d registros", len(df))
        return df