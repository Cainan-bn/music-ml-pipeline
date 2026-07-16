

class PlaycatchDomainError(Exception):


class DataSourceError(PlaycatchDomainError):


class SentimentAnalysisError(PlaycatchDomainError):


class ModelNotFoundError(PlaycatchDomainError):


class FeatureExtractionError(PlaycatchDomainError):


class InferenceError(PlaycatchDomainError):


class SchemaValidationError(PlaycatchDomainError):

class ChatbotError(PlaycatchDomainError):
