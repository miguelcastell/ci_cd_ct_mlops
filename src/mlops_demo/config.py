"""Configuração central do projeto: caminhos, nomes de colunas e limiares de qualidade."""

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]

# dados
DADOS_FONTE = RAIZ / "data" / "fonte" / "spambase.data"  # arquivo original da UCI, sem cabeçalho
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_LOTES = DADOS_BRUTOS / "lotes"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
BASE_TREINO = DADOS_BRUTOS / "base_treino.csv"
BASE_HOLDOUT = DADOS_BRUTOS / "base_holdout.csv"
TREINO_PROCESSADO = DADOS_PROCESSADOS / "treino.csv"
AVALIACAO_PROCESSADA = DADOS_PROCESSADOS / "avaliacao.csv"

# modelos e relatórios
MODELO_CAMPEAO = RAIZ / "models" / "champion" / "model.joblib"
METRICAS_CAMPEAO = RAIZ / "models" / "champion" / "metrics.json"
MODELO_CANDIDATO = RAIZ / "models" / "candidate" / "model.joblib"
METRICAS_CANDIDATO = RAIZ / "reports" / "candidate_metrics.json"
RELATORIO_GATE = RAIZ / "reports" / "gate.json"
RESUMO_GATE = RAIZ / "reports" / "gate.md"
RELATORIO_DRIFT = RAIZ / "reports" / "drift.md"
PARAMS = RAIZ / "params.yaml"

# esquema dos dados (Spambase, UCI). A ordem é a do arquivo original.
# word_freq_*: % das palavras do e-mail iguais à palavra (0 a 100)
# char_freq_*: % dos caracteres do e-mail iguais ao caractere (0 a 100)
# capital_run_length_*: média, maior e total das sequências de letras MAIÚSCULAS (>= 1)
PALAVRAS = [
    "make", "address", "all", "3d", "our", "over", "remove", "internet", "order", "mail",
    "receive", "will", "people", "report", "addresses", "free", "business", "email", "you",
    "credit", "your", "font", "000", "money", "hp", "hpl", "george", "650", "lab", "labs",
    "telnet", "857", "data", "415", "85", "technology", "1999", "parts", "pm", "direct", "cs",
    "meeting", "original", "project", "re", "edu", "table", "conference",
]  # fmt: skip
# os caracteres originais (; ( [ ! $ #) viram nomes legíveis para a API e o CSV
CARACTERES = ["semicolon", "paren", "bracket", "exclamation", "dollar", "hash"]
COLUNAS_FREQUENCIA = [f"word_freq_{p}" for p in PALAVRAS] + [f"char_freq_{c}" for c in CARACTERES]
COLUNAS_MAIUSCULAS = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]
FEATURES = COLUNAS_FREQUENCIA + COLUNAS_MAIUSCULAS  # 57 atributos
ALVO = "spam"
CLASSE_SPAM = 1  # classe positiva: 1 = spam, 0 = e-mail legítimo

# Sinais clássicos de spam: usados para simular drift e no teste direcional
# (mais sinais de spam não podem reduzir a probabilidade de spam).
COLUNAS_SINAL_SPAM = [
    "word_freq_free",
    "word_freq_remove",
    "word_freq_money",
    "char_freq_exclamation",
    "char_freq_dollar",
    *COLUNAS_MAIUSCULAS,
]
# Palavras que, neste dataset, indicam e-mail legítimo (o doador trabalhava na HP, se chama George)
COLUNAS_SINAL_LEGITIMO = ["word_freq_george", "word_freq_hp", "word_freq_hpl"]

# Limiares de qualidade (o "contrato" do modelo). Mudar aqui exige revisão em PR.
# Erro mais caro: FALSO POSITIVO (e-mail legítimo jogado no spam e nunca lido).
PRECISAO_MINIMA = 0.93  # métrica principal: no máximo 7% do que vai para o spam é legítimo
RECALL_MINIMO = 0.85  # sem ele, um modelo que quase nunca diz "spam" teria precisão alta
TOLERANCIA_CAMPEAO = 0.0  # o candidato não pode ter precisão menor que o campeão
PISO_SANIDADE = 0.80  # o campeão não pode colapsar na base de referência (holdout fixo)
LIMITE_DRIFT_P = 0.01  # p-valor do teste KS
MIN_FEATURES_DRIFT = 4  # gatilho do CT: atributos com drift (de 57) que disparam o retreino


def carregar_params() -> dict:
    """Lê o params.yaml (hiperparâmetros e semente)."""
    with open(PARAMS, encoding="utf-8") as arquivo:
        return yaml.safe_load(arquivo)
