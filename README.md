# Sistema de Recomendação Musical por Sentimento

Pipeline de Machine Learning que analisa o sentimento de letras de música e
recomenda faixas com base no humor do usuário, integrado a um chatbot
conversacional (Gemini) para descoberta musical em linguagem natural.

## Arquitetura

Clean Architecture com inversão de dependência: o domínio (`domain/`) não
conhece frameworks de ML nem provedores de LLM. Implementações concretas
(`infrastructure/`) são injetadas nos casos de uso (`application/`) via
construtor, atrás de interfaces (`ABC`/`Protocol`).

```
domain/           entidades, exceções, contratos (interfaces)
infrastructure/    fontes de dados, modelos, LLM client, factories
application/       estratégias (features, validação), pipelines, avaliação
chatbot/           motor conversacional e orquestração de diálogo
```

**Padrões aplicados:**

- **Factory** — `SentimentAnalyzerFactory`, `ModelFactory`: criação
  desacoplada de componentes por tipo configurado, sem `if/else` espalhado
  pelo código cliente.
- **Strategy** — `IFeatureExtractor` (TF-IDF), `IValidationStrategy`
  (KFold / holdout): trocáveis sem alterar os pipelines que os consomem.
- **Dependency Inversion** — `TrainingPipeline`, `InferencePipeline` e
  `ConversationService` dependem apenas de interfaces, nunca de classes
  concretas de infraestrutura.
- **Interface Segregation** — `IProbabilisticClassifier` isola o contrato
  usado apenas pela avaliação (`predict_proba`), sem forçar todo modelo de
  produção a implementá-lo.

## Componentes de ML

| Componente | Implementação | Interface |
|---|---|---|
| Sentimento | `GroundTruthSentimentAnalyzer` (dataset rotulado) / `HuggingFaceSentimentAnalyzer` (heurística plugável) | `ISentimentAnalyzer` |
| Features | TF-IDF (`TfidfFeatureStrategy`) | `IFeatureExtractor` |
| Recomendação | LightGBM multiclasse, ranking por probabilidade da emoção-alvo | `IRecommenderModel` |
| Chatbot | Gemini via `ChatbotEngine` — classifica emoção da mensagem e gera resposta | `IChatbotEngine` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  
```

## Uso

```bash
# 1. Gerar dataset sintético de demonstração
python scripts/generate_synthetic_dataset.py

# 2. Treinar o pipeline (sentimento + recomendação)
python scripts/run_training.py

# 3. Avaliar o modelo por validação cruzada
python scripts/evaluate_model.py

# 4. Gerar recomendações
python scripts/run_inference.py

# 5. Conversar com o chatbot (requer chave Gemini válida)
python scripts/run_chatbot_demo.py
```

## Testes

```bash
pytest -v
```

21 testes (unitários + integração), cobrindo cada camada isoladamente via
test doubles (fakes/dummies). Nenhum teste depende de rede ou de uma API de
LLM real.

## Avaliação do modelo

Validação cruzada em 5 folds sobre o dataset sintético (150 amostras, 6
classes):

```
Acurácia média: 88.67%
Por fold: 90.0% / 90.0% / 83.3% / 90.0% / 90.0%
```

O vocabulário do gerador sintético foi desenhado com sobreposição
intencional entre emoções (ex: "night", "heart", "time" aparecem em
múltiplas categorias), para evitar classes trivialmente separáveis e
produzir uma métrica de avaliação realista.

## Limitações conhecidas

- **Dataset sintético**: as letras usadas em `generate_synthetic_dataset.py`
  são geradas proceduralmente para fins de demonstração da arquitetura, não
  refletem a diversidade linguística de letras reais. Métricas de avaliação
  aqui **não devem ser extrapoladas** para desempenho em produção.
- **TF-IDF**: representação bag-of-words; não captura ordem ou semântica
  profunda. Uma evolução natural seria embeddings (ex: sentence-transformers).
- **Validação do chatbot**: o `ConversationService` usa fallback (`calm`)
  quando a classificação de emoção via LLM falha, mas não há retry
  configurável.

## Próximos passos

- Substituir o dataset sintético por um corpus real de letras rotuladas.
- Adicionar embeddings como `IFeatureExtractor` alternativo (Strategy).
- Expor `InferencePipeline` via API REST (FastAPI) para consumo externo.
- Métricas de negócio (ex: taxa de cliques nas recomendações) além de acurácia.

## Stack

Python 3.11+, pandas, scikit-learn, LightGBM, google-generativeai, pytest.
