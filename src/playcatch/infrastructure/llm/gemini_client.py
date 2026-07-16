
from google import genai

from playcatch.config.logging_config import get_logger
from playcatch.chatbot.interfaces import ILLMClient
from playcatch.domain.exceptions import ChatbotError

logger = get_logger(__name__)


class GeminiLLMClient(ILLMClient):

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
            if not api_key:
            raise ChatbotError(
                "GEMINI_API_KEY não configurada. Defina PLAYCATCH_GEMINI_API_KEY no .env"
            )
        
        # O novo SDK utiliza o cliente instanciado em vez de configuração global
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(self, prompt: str) -> str:
            try:
            # Chamada unificada do SDK moderno
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Falha na chamada à API Gemini: %s", exc)
            raise ChatbotError(f"Falha na chamada ao Gemini: {exc}") from exc

        if not response.text:
            raise ChatbotError("Gemini retornou resposta vazia")

        return response.text.strip()