
import pandas as pd
import pytest

from playcatch.domain.entities import Lyrics


@pytest.fixture
def sample_lyrics() -> Lyrics:
    return Lyrics(song_id="song_001", text="Hoje é um dia feliz e ensolarado")


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "song_id": ["s1", "s2"],
            "title": ["Música A", "Música B"],
            "artist": ["Artista A", "Artista B"],
            "lyrics": ["Estou muito feliz hoje", "Que tristeza profunda"],
        }
    )