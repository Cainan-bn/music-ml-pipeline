
import random
from pathlib import Path

import pandas as pd

random.seed(42)

OUTPUT_PATH = Path("data/raw/songs_lyrics.csv")

PHRASE_BANK: dict[str, list[str]] = {
    "happy": [
        "the sun is shining bright and my heart feels light tonight",
        "we are dancing in the street with joy running through us",
        "laughter fills the night like a warm and easy feeling",
        "every smile today feels like a quiet little victory",
        "the whole world feels golden when I think of this time",
        "there is a lightness in my chest I cannot explain",
    ],
    "sad": [
        "the rain keeps falling on this quiet empty night",
        "I miss the way things felt before you found the door",
        "there is a heavy silence where your voice used to be",
        "the photographs are fading like the time we used to share",
        "walking home alone feels longer every single night",
        "my heart still aches in places I thought were healed",
    ],
    "angry": [
        "I am done staying quiet while you break every promise",
        "the fire in my chest will not go out tonight",
        "stop pretending you did not see this coming for miles",
        "my patience ran out somewhere between the lies and the time",
        "this is the last night I let you walk all over me",
        "there is a heat in my heart that feels like war",
    ],
    "calm": [
        "the lake is still and the night air feels soft",
        "slow breaths carry me into a peaceful quiet time",
        "the candle flickers gently as the night settles in",
        "there is nothing to chase here, only stillness and rest",
        "my heart slows down when the world goes quiet like this",
        "the mountains hold their silence like an old and steady friend",
    ],
    "energetic": [
        "the beat drops and the whole night jumps to life",
        "we are running full speed with the wind at our backs",
        "turn it up loud because tonight has no limits",
        "the crowd roars as the countdown hits the final second",
        "my heart is racing faster than this feeling can explain",
        "adrenaline runs through every step of this wild night",
    ],
    "romantic": [
        "your hand in mine feels like the safest place tonight",
        "every letter I write somehow always comes back to you",
        "the way you look at me still makes my heart feel light",
        "I would cross any distance just to share this time",
        "under the stars, it is only ever the two of us tonight",
        "my heart still skips whenever you walk into the room",
    ],
}

FICTIONAL_ARTISTS = [
    "The Paper Owls", "Velvet Horizon", "Neon Static", "Coastal Drift",
    "Midnight Foxes", "Glass Avenue", "Amber Skyline", "The Quiet Signal",
]


def build_lyrics(emotion: str) -> str:
    fragments = random.sample(PHRASE_BANK[emotion], k=2)
    return ". ".join(fragment.capitalize() for fragment in fragments) + "."


def main(rows_per_emotion: int = 25) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    song_counter = 1
    for emotion in PHRASE_BANK:
        for i in range(rows_per_emotion):
            records.append(
                {
                    "song_id": f"s{song_counter}",
                    "title": f"{emotion.capitalize()} Track {i + 1}",
                    "artist": random.choice(FICTIONAL_ARTISTS),
                    "lyrics": build_lyrics(emotion),
                    "emotion": emotion,
                }
            )
            song_counter += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Dataset sintético com {len(df)} músicas salvo em '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()