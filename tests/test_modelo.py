"""Testes de modelo: limiares, baseline e comportamento (ML Test Score, Breck et al., 2017)."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from mlops_demo import config
from mlops_demo.data import aplicar_drift
from mlops_demo.evaluate import avaliar, calcular_metricas
from mlops_demo.features import separar_xy
from mlops_demo.train import treinar

pytestmark = pytest.mark.modelo


def test_campeao_versionado_foi_certificado_pelo_portao():
    """O metrics.json do campeão registra o que o portão mediu ao aprová-lo e cumpre o contrato."""
    with open(config.METRICAS_CAMPEAO, encoding="utf-8") as arquivo:
        declaradas = json.load(arquivo)
    assert declaradas["precisao_spam"] >= config.PRECISAO_MINIMA, declaradas
    assert declaradas["recall_spam"] >= config.RECALL_MINIMO, declaradas


def test_campeao_versionado_nao_colapsou_na_base_de_referencia(campeao, base_holdout):
    """Sanidade: o artefato carrega, prevê e não perdeu o básico no holdout fixo.

    O contrato (precisão 0,93 e recall 0,85) é medido pelo portão, no conjunto de avaliação da
    época da promoção.
    Depois que o modelo se adapta a dados novos, o holdout antigo pode cair um pouco.
    Por isso o piso aqui é menor.
    """
    metricas = avaliar(campeao, base_holdout)
    assert metricas["precisao_spam"] >= config.PISO_SANIDADE, metricas
    assert metricas["recall_spam"] >= config.PISO_SANIDADE, metricas


def test_treino_com_o_codigo_e_params_atuais_cumpre_o_contrato(modelo_atual, base_holdout):
    """Se alguém mexer no código ou no params.yaml, o CI treina de novo e confere os limiares."""
    metricas = avaliar(modelo_atual, base_holdout)
    assert metricas["precisao_spam"] >= config.PRECISAO_MINIMA, metricas
    assert metricas["recall_spam"] >= config.RECALL_MINIMO, metricas


def test_modelo_supera_o_baseline_ingenuo(modelo_atual, base_treino, base_holdout):
    X_tr, y_tr = separar_xy(base_treino)
    X_te, y_te = separar_xy(base_holdout)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    nota_baseline = calcular_metricas(y_te, baseline.predict(X_te))["acuracia"]
    nota_modelo = calcular_metricas(y_te, modelo_atual.predict(X_te))["acuracia"]
    assert nota_modelo > nota_baseline + 0.1


def test_treino_e_deterministico(base_treino, params, xy_holdout):
    X_te, _ = xy_holdout
    a = treinar(base_treino, params).predict_proba(X_te)
    b = treinar(base_treino, params).predict_proba(X_te)
    assert np.allclose(a, b)


def test_modelo_consegue_memorizar_um_lote_pequeno(base_treino, params):
    """Teste clássico de sanidade: se nem um lote pequeno é aprendido, há um bug no pipeline.

    Com 57 atributos e só 40 linhas, a regularização do params.yaml (C=0.1) impede a
    memorização de propósito. Aqui o C é afrouxado para medir só a capacidade do pipeline.
    """
    pequeno = base_treino.sample(40, random_state=0)
    sem_regularizacao = {**params, "modelo": {**params["modelo"], "C": 100.0}}
    modelo = treinar(pequeno, sem_regularizacao)
    X, y = separar_xy(pequeno)
    assert (modelo.predict(X) == y).mean() >= 0.95


def test_probabilidades_sao_validas(modelo_atual, xy_holdout):
    X_te, _ = xy_holdout
    probas = modelo_atual.predict_proba(X_te)
    assert np.all((probas >= 0) & (probas <= 1))
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_invariancia_a_ruido_minimo(modelo_atual, xy_holdout):
    """Teste de comportamento: ruído de 0,01% nos atributos quase nunca muda a previsão."""
    X_te, _ = xy_holdout
    rng = np.random.default_rng(0)
    ruidoso = X_te * (1 + rng.normal(0, 1e-4, X_te.shape))
    mudou = (modelo_atual.predict(X_te) != modelo_atual.predict(ruidoso)).mean()
    assert mudou <= 0.02


def _p_spam(modelo, X):
    coluna = list(modelo.classes_).index(config.CLASSE_SPAM)
    return modelo.predict_proba(X)[:, coluna]


def test_direcional_mais_sinais_de_spam_nao_reduz_a_chance_de_spam(modelo_atual, xy_holdout):
    """Teste direcional: mais "free", "remove", "money", "!", "$" e MAIÚSCULAS não podem
    reduzir a probabilidade média de spam."""
    X_te, _ = xy_holdout
    mais_spam = aplicar_drift(X_te.assign(**{config.ALVO: 0}), 1.5)[config.FEATURES]
    assert _p_spam(modelo_atual, mais_spam).mean() >= _p_spam(modelo_atual, X_te).mean()


def test_direcional_palavras_do_destinatario_nao_aumentam_a_chance_de_spam(
    modelo_atual, xy_holdout
):
    """Novo (específico do problema): no Spambase, "george", "hp" e "hpl" vêm do dono da caixa
    (George, da HP Labs). Citar mais essas palavras não pode deixar o e-mail mais "spam"."""
    X_te, _ = xy_holdout
    pessoal = X_te.copy()
    for c in config.COLUNAS_SINAL_LEGITIMO:
        pessoal[c] = (pessoal[c] + 1.0).clip(upper=100.0)
    assert _p_spam(modelo_atual, pessoal).mean() <= _p_spam(modelo_atual, X_te).mean()


def test_precisao_por_classe_nao_esconde_falsos_positivos(modelo_atual, xy_holdout):
    """Novo: no máximo 5% dos e-mails LEGÍTIMOS do holdout podem ir para o spam (taxa de FP)."""
    X_te, y_te = xy_holdout
    legitimos = y_te != config.CLASSE_SPAM
    taxa_fp = (modelo_atual.predict(X_te[legitimos]) == config.CLASSE_SPAM).mean()
    assert taxa_fp <= 0.05, taxa_fp
