"""Testes de dados: esquema, valores do domínio e vazamento entre treino e teste."""

import pandas as pd
import pytest

from mlops_demo import config
from mlops_demo.data import listar_lotes, validar_dataframe

pytestmark = pytest.mark.dados


def _arquivos_de_dados():
    return [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]


@pytest.mark.parametrize("caminho", _arquivos_de_dados(), ids=lambda c: c.name)
def test_arquivo_respeita_esquema_e_valores(caminho):
    problemas = validar_dataframe(pd.read_csv(caminho))
    assert problemas == [], f"{caminho.name}: {problemas}"


def test_base_de_treino_tem_as_duas_classes_em_proporcao_razoavel(base_treino):
    # no Spambase original ~39% dos e-mails são spam
    proporcao_spam = (base_treino[config.ALVO] == config.CLASSE_SPAM).mean()
    assert 0.30 <= proporcao_spam <= 0.50


def test_nao_ha_vazamento_entre_treino_e_holdout(base_treino, base_holdout):
    chaves_treino = set(map(tuple, base_treino[config.FEATURES].round(6).to_numpy()))
    chaves_holdout = set(map(tuple, base_holdout[config.FEATURES].round(6).to_numpy()))
    assert chaves_treino.isdisjoint(chaves_holdout)


def test_nao_ha_linhas_duplicadas_na_base_de_treino(base_treino):
    assert not base_treino.duplicated().any()


def test_validador_detecta_nulos_e_valores_negativos(base_treino):
    corrompido = base_treino.copy()
    corrompido.loc[0, "word_freq_free"] = None
    corrompido.loc[1, "char_freq_dollar"] = -5.0
    problemas = validar_dataframe(corrompido)
    assert any("nulos" in p for p in problemas)
    assert any("negativos" in p for p in problemas)


def test_validador_detecta_esquema_diferente(base_treino):
    assert validar_dataframe(base_treino.drop(columns=["word_freq_free"]))


def test_validador_detecta_frequencia_acima_de_100_por_cento(base_treino):
    # novo (específico do problema): frequência é percentual, 150% é impossível
    corrompido = base_treino.copy()
    corrompido.loc[0, "word_freq_money"] = 150.0
    assert any("acima de 100%" in p for p in validar_dataframe(corrompido))


def test_validador_detecta_maiusculas_incoerentes(base_treino):
    # novo: a maior sequência de maiúsculas não pode ser maior que o total de maiúsculas
    corrompido = base_treino.copy()
    corrompido.loc[0, "capital_run_length_longest"] = (
        corrompido.loc[0, "capital_run_length_total"] + 1
    )
    assert any("incoerentes" in p for p in validar_dataframe(corrompido))


def test_validador_detecta_contagem_de_maiusculas_nao_inteira(base_treino):
    corrompido = base_treino.copy()
    corrompido["capital_run_length_total"] = corrompido["capital_run_length_total"].astype(float)
    corrompido.loc[0, "capital_run_length_total"] += 0.5
    assert any("não inteiras" in p for p in validar_dataframe(corrompido))
