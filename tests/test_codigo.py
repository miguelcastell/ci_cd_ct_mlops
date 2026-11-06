"""Testes de código: funções e pipeline se comportam como o contrato diz."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlops_demo import config
from mlops_demo.data import aplicar_drift, validar_dataframe
from mlops_demo.evaluate import calcular_metricas
from mlops_demo.features import construir_pipeline, separar_xy
from mlops_demo.gate import decidir

pytestmark = pytest.mark.codigo


def test_separar_xy_devolve_57_atributos_e_alvo(base_treino):
    X, y = separar_xy(base_treino)
    assert X.shape == (len(base_treino), 57)
    assert len(y) == len(base_treino)


def test_separar_xy_impoe_a_ordem_das_colunas(base_treino):
    embaralhado = base_treino[list(reversed(base_treino.columns))]
    X, _ = separar_xy(embaralhado)
    assert list(X.columns) == config.FEATURES


def test_separar_xy_falha_quando_falta_coluna(base_treino):
    sem_coluna = base_treino.drop(columns=["word_freq_free"])
    with pytest.raises(ValueError, match="Colunas ausentes"):
        separar_xy(sem_coluna)


def test_separar_xy_nao_altera_o_dataframe_original(base_treino):
    antes = base_treino.copy()
    separar_xy(base_treino)
    pd.testing.assert_frame_equal(base_treino, antes)


def test_pipeline_tem_log_padronizacao_e_classificador(params):
    pipeline = construir_pipeline(params)
    assert isinstance(pipeline, Pipeline)
    assert [nome for nome, _ in pipeline.steps] == ["log1p", "padronizar", "classificador"]


def test_metricas_usam_spam_como_classe_positiva():
    # modelo que marca tudo como spam (1): recall 1.0 e precisão = proporção real de spam
    y_real = [1, 1, 0, 0]
    metricas = calcular_metricas(y_real, [1, 1, 1, 1])
    assert metricas["recall_spam"] == 1.0
    assert metricas["precisao_spam"] == pytest.approx(0.5)


def test_metricas_nao_quebram_quando_o_modelo_nunca_diz_spam():
    # novo (específico do problema): filtro "desligado" tem precisão 0 por definição, sem erro
    metricas = calcular_metricas([1, 0, 0], [0, 0, 0])
    assert metricas["precisao_spam"] == 0.0
    assert metricas["recall_spam"] == 0.0


def test_drift_simulado_mantem_os_dados_validos_no_dominio(base_holdout):
    # novo (específico do problema): o drift não pode gerar % > 100 nem maiúsculas incoerentes
    for fator in (0.5, 2.0, 50.0):
        assert validar_dataframe(aplicar_drift(base_holdout, fator)) == [], fator


def test_drift_simulado_so_mexe_nos_sinais_de_spam(base_holdout):
    com_drift = aplicar_drift(base_holdout, 2.0)
    intactas = [c for c in config.FEATURES if c not in config.COLUNAS_SINAL_SPAM]
    pd.testing.assert_frame_equal(com_drift[intactas], base_holdout[intactas])
    assert (com_drift["char_freq_exclamation"] >= base_holdout["char_freq_exclamation"]).all()


# --- o próprio portão de qualidade precisa ser testado ---------------------------------------


def _metricas(precisao, recall):
    return {"precisao_spam": precisao, "recall_spam": recall, "n_amostras": 100}


def test_gate_aprova_candidato_melhor_e_dentro_dos_limiares():
    assert decidir(_metricas(0.96, 0.90), _metricas(0.95, 0.90))["aprovado"]


def test_gate_reprova_precisao_abaixo_do_limiar():
    # recall quase perfeito não compensa mandar e-mail legítimo para o spam
    assert not decidir(_metricas(0.90, 0.99), _metricas(0.85, 0.90))["aprovado"]


def test_gate_reprova_recall_abaixo_do_limiar():
    # filtro "medroso": precisão perfeita, mas deixa passar spam demais
    assert not decidir(_metricas(1.0, 0.60), _metricas(0.95, 0.90))["aprovado"]


def test_gate_reprova_candidato_com_precisao_pior_que_o_campeao():
    assert not decidir(_metricas(0.94, 0.95), _metricas(0.96, 0.88))["aprovado"]
