"""Pré-processamento e construção do pipeline de modelo."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from mlops_demo.config import ALVO, FEATURES


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa atributos e alvo, garantindo as colunas esperadas e a ordem."""
    faltando = [c for c in FEATURES + [ALVO] if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes: {faltando}")
    X = df[FEATURES[:-1]].copy()
    y = df[ALVO].astype(int)
    return X, y


def construir_pipeline(params: dict) -> Pipeline:
    """Pipeline de treino: log(1 + x) + padronização + regressão logística.

    As frequências e as contagens de maiúsculas do Spambase são muito assimétricas (muitos zeros
    e caudas longas). O log1p comprime as caudas antes da padronização.
    """
    modelo = params["modelo"]
    return Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p)),
            ("padronizar", StandardScaler()),
            (
                "classificador",
                LogisticRegression(
                    C=modelo["C"],
                    max_iter=modelo["max_iter"],
                    class_weight=modelo["class_weight"],
                    random_state=params["semente"],
                ),
            ),
        ]
    )
