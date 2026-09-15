# prj_churn · Curso de MLOps — IESB

Projeto-fio-condutor do curso: previsão de **churn** (classificação binária tabular).  
O arco é do `train_churn.py` monolítico até um sistema completo de MLOps.

---

## Estrutura

```
prj_churn/
├─ src/churn/
│  ├─ config.py        # Settings (Pydantic) — caminhos, hiperparâmetros
│  ├─ data.py          # carga, limpeza, validação (pandera)
│  ├─ schema.py        # ChurnSchema — contrato de dados
│  ├─ features.py      # transformações puras + encoder sem vazamento
│  ├─ model.py         # treino + MLflow tracking
│  ├─ evaluate.py      # métricas
│  ├─ experiments.py   # grid rastreado (MLflow)
│  └─ lineage.py       # data_md5() e git_commit()
├─ tests/              # pytest (data, features, schema)
├─ data/
│  ├─ churn.csv.dvc    # ponteiro DVC (o .csv fica no remote)
│  └─ .gitignore
├─ Dockerfile          # imagem com uv (enc. 3)
├─ compose.yml         # docker compose (enc. 3)
├─ dvc.yaml            # pipeline reproduzível (enc. 5)
└─ pyproject.toml      # dependências travadas (uv)
```

---

## Bloco 0 — Fundação (Encontros 1–3)

### Rodando localmente

```bash
uv sync                              # cria o ambiente a partir do pyproject
uv run python -m churn.model         # treina e imprime métricas
uv run pytest                        # testa
```

### Rodando no container (Docker)

```bash
docker build -t churn:0.1 .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts churn:0.1
```

Ou com compose:

```bash
docker compose up treino
```

---

## Bloco 1 — Data Pipeline (Encontros 4–5)

### Encontro 4 — Pandera

O `data.py` valida o dataset antes do treino com o `ChurnSchema` (pandera).  
Dado ruim para o pipeline **cedo e com barulho**, não silenciosamente após a métrica cair.

```python
# para forçar falha e ver o contrato em ação:
from churn.schema import ChurnSchema
ChurnSchema.validate(df_ruim, lazy=True)  # SchemaErrors com todas as violações
```

### Encontro 5 — DVC

O `churn.csv` não vai para o Git — só o ponteiro `churn.csv.dvc` vai.  
Os bytes ficam em um remote (local ou S3).

```bash
dvc pull           # baixa o dado do remote
dvc repro          # re-executa só o que mudou (dado ou código)
dvc push           # sobe a nova versão do dado
```

---

## Bloco 2 — ML Pipeline (Encontro 6)

### MLflow Tracking

```bash
uv run python -m churn.model          # um run rastreado
uv run python -m churn.experiments    # grid: 9 combinações (n_estimators × max_depth)
uv run mlflow ui                      # UI em localhost:5000
```

Achar o melhor run por código:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["churn"], order_by=["metrics.roc_auc DESC"])
print(runs.iloc[0][["params.n_estimators", "params.max_depth", "metrics.roc_auc"]])
```

---

## O que mudou em relação ao `train_churn.py`

| Script monolítico | Projeto modular |
|---|---|
| caminho hardcoded `C:/Users/ana/...` | `config.py` (Pydantic, sobrescrevível por env) |
| `fillna(2200)` mágico | `TotalCharges` em branco → `0.0` (cliente novo) |
| normalização por números mágicos | removida — Random Forest não precisa de escala |
| `LabelEncoder` no dataset inteiro (vazamento) | `CategoricalEncoder` com `fit` só no treino |
| sem `random_state` (acurácia instável) | semente fixa em split e modelo |
| só `accuracy` | 5 métricas: accuracy, precision, recall, f1, roc_auc |
| sem validação de dados | `ChurnSchema` (pandera) — falha cedo e com mensagem |
| dataset solto no disco | `churn.csv.dvc` — versionado pelo DVC |
| `print(accuracy)` e esquece | MLflow registra params, métricas e artefatos |

## Próximos encontros

- **Enc. 7** — Model Registry (MLflow).
- **Enc. 8** — Empacotamento com MLflow Models.
- **Enc. 9–11** — Orquestração com Airflow.
- **Enc. 12–14** — Serving com FastAPI + CI/CD.
- **Enc. 15** — Monitoramento com Evidently.
