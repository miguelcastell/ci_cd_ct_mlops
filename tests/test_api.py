"""Testes de contrato da API: formato de entrada e de saída do /predict."""

import pytest
from fastapi.testclient import TestClient

from mlops_demo import config
from mlops_demo.api import app

pytestmark = pytest.mark.api

cliente = TestClient(app)


def test_health_responde_ok():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_predict_devolve_o_formato_combinado(base_holdout):
    email = base_holdout[config.FEATURES].iloc[0].tolist()
    resposta = cliente.post("/predict", json={"atributos": email})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"classe", "probabilidade_spam", "versao_codigo"}
    assert corpo["classe"] in {"spam", "legitimo"}
    assert 0.0 <= corpo["probabilidade_spam"] <= 1.0


def test_predict_rejeita_quantidade_errada_de_atributos():
    resposta = cliente.post("/predict", json={"atributos": [1.0, 2.0]})
    assert resposta.status_code == 422


def test_predict_acerta_um_spam_e_um_legitimo_claros(base_holdout, campeao):
    """Escolhe os casos mais claros (maior confiança do campeão) de cada classe."""
    X = base_holdout[config.FEATURES]
    coluna = list(campeao.classes_).index(config.CLASSE_SPAM)
    p_spam = campeao.predict_proba(X)[:, coluna]
    for indice, esperado in ((int(p_spam.argmax()), "spam"), (int(p_spam.argmin()), "legitimo")):
        resposta = cliente.post("/predict", json={"atributos": X.iloc[indice].tolist()})
        assert resposta.json()["classe"] == esperado


def test_predict_rejeita_frequencia_negativa(base_holdout):
    # novo (específico do problema): frequência de palavra negativa não existe
    email = base_holdout[config.FEATURES].iloc[0].tolist()
    email[config.FEATURES.index("word_freq_free")] = -1.0
    assert cliente.post("/predict", json={"atributos": email}).status_code == 422


def test_atributos_informa_a_ordem_esperada():
    corpo = cliente.get("/atributos").json()
    assert corpo["n_atributos"] == 57
    assert corpo["atributos"] == config.FEATURES


def test_email_tipico_de_spam_e_classificado_como_spam():
    """Novo: e-mail sintético com "free", "money", "remove", "!", "$" e muitas MAIÚSCULAS."""
    email = dict.fromkeys(config.FEATURES, 0.0)
    email.update(
        word_freq_free=3.0, word_freq_money=2.0, word_freq_remove=2.0, word_freq_your=3.0,
        word_freq_000=1.5, char_freq_exclamation=1.5, char_freq_dollar=0.8,
        capital_run_length_average=8.0, capital_run_length_longest=60.0,
        capital_run_length_total=600.0,
    )  # fmt: skip
    corpo = cliente.post("/predict", json={"atributos": list(email.values())}).json()
    assert corpo["classe"] == "spam", corpo
