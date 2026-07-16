
from playcatch.domain.exceptions import ModelNotFoundError
from playcatch.domain.interfaces import IRecommenderModel
from playcatch.infrastructure.models.lightgbm_recommender import LightGbmRecommender


class ModelFactory:

    _registry: dict[str, type[IRecommenderModel]] = {
        "lightgbm": LightGbmRecommender,
    }

    @classmethod
    def create(cls, model_type: str, **kwargs: object) -> IRecommenderModel:
            model_cls = cls._registry.get(model_type)
        if model_cls is None:
            raise ModelNotFoundError(
                f"Modelo '{model_type}' não registrado. "
                f"Opções disponíveis: {list(cls._registry.keys())}"
            )
        return model_cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def register(cls, name: str, model_cls: type[IRecommenderModel]) -> None:
            cls._registry[name] = model_cls