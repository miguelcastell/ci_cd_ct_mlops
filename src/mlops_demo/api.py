"""API de inferência (FastAPI) servindo o modelo campeão do filtro de spam.

Local:   uvicorn mlops_demo.api:app --reload
"""

import json
import os
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from mlops_demo import __version__, config

CAMINHO_MODELO = Path(os.environ.get("MODEL_PATH", config.MODELO_CAMPEAO))
CAMINHO_METRICAS = CAMINHO_MODELO.with_name("metrics.json")
N_ATRIBUTOS = len(config.FEATURES)  # 57

app = FastAPI(title="Filtro de spam (demo CI/CD/CT)", version=__version__)
_modelo = joblib.load(CAMINHO_MODELO)


class Email(BaseModel):
    """Os 57 atributos do e-mail (Spambase), na mesma ordem do treino (GET /atributos).

    48 frequências de palavras e 6 de caracteres (% de 0 a 100), seguidas das 3 medidas de
    sequências de letras maiúsculas (média, maior e total, todas >= 1).
    """

    atributos: list[Annotated[float, Field(ge=0)]] = Field(
        min_length=N_ATRIBUTOS, max_length=N_ATRIBUTOS
    )


class Predicao(BaseModel):
    classe: str  # "spam" ou "legitimo"
    probabilidade_spam: float
    versao_codigo: str


@app.get("/health")
def saude() -> dict:
    metricas = {}
    if CAMINHO_METRICAS.exists():
        metricas = json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))
    return {"status": "ok", "versao_codigo": __version__, "metricas_do_campeao": metricas}


@app.get("/atributos")
def atributos() -> dict:
    """Ordem esperada dos atributos em POST /predict."""
    return {"n_atributos": N_ATRIBUTOS, "atributos": config.FEATURES}


@app.post("/predict", response_model=Predicao)
def prever(email: Email) -> Predicao:
    entrada = pd.DataFrame([email.atributos], columns=config.FEATURES)
    previsto = int(_modelo.predict(entrada)[0])
    # a coluna da classe spam (1) em predict_proba segue a ordem de classes_
    coluna = list(_modelo.classes_).index(config.CLASSE_SPAM)
    p_spam = float(_modelo.predict_proba(entrada)[0][coluna])
    return Predicao(
        classe="spam" if previsto == config.CLASSE_SPAM else "legitimo",
        probabilidade_spam=round(p_spam, 4),
        versao_codigo=__version__,
    )
