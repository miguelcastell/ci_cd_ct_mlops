"""Dados: criação da base, validação de esquema, geração de lotes e preparação.

Uso na linha de comando:
    python -m mlops_demo.data base      # cria base_treino.csv e base_holdout.csv (uma vez)
    python -m mlops_demo.data preparar  # junta base + lotes e gera data/processed/
    python -m mlops_demo.data validar   # valida todos os CSVs de data/raw/
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_demo import config

SEMENTE_DADOS = 7  # divisão fixa da base: não mude, ou o campeão perde a referência
N_TREINO_BASE = 2000
N_HOLDOUT_BASE = 1000


def carregar_dataset_completo() -> pd.DataFrame:
    """Spambase (UCI) sem duplicatas: 4601 linhas no original, 4204 após a limpeza.

    Remove e-mails com os mesmos atributos (duplicatas), e também os que aparecem com rótulos
    diferentes, para que treino, holdout e lotes nunca compartilhem uma linha.
    """
    df = pd.read_csv(config.DADOS_FONTE, header=None, names=[*config.FEATURES, config.ALVO])
    rotulos_por_linha = df.groupby(config.FEATURES)[config.ALVO].transform("nunique")
    df = df[rotulos_por_linha == 1]
    return df.drop_duplicates(subset=config.FEATURES).reset_index(drop=True)


def dividir_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em base de treino, holdout fixo e reserva (origem dos lotes novos)."""
    df = carregar_dataset_completo()
    resto, base_treino = train_test_split(
        df, test_size=N_TREINO_BASE, stratify=df[config.ALVO], random_state=SEMENTE_DADOS
    )
    reserva, base_holdout = train_test_split(
        resto, test_size=N_HOLDOUT_BASE, stratify=resto[config.ALVO], random_state=SEMENTE_DADOS
    )
    return (
        base_treino.reset_index(drop=True),
        base_holdout.reset_index(drop=True),
        reserva.reset_index(drop=True),
    )


def criar_base() -> None:
    """Grava a base de treino e o holdout fixo em data/raw/."""
    base_treino, base_holdout, _ = dividir_dataset()
    config.DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    base_treino.to_csv(config.BASE_TREINO, index=False, lineterminator="\n")
    base_holdout.to_csv(config.BASE_HOLDOUT, index=False, lineterminator="\n")
    print(f"base_treino: {len(base_treino)} linhas | base_holdout: {len(base_holdout)} linhas")


def aplicar_drift(df: pd.DataFrame, fator: float) -> pd.DataFrame:
    """Simula uma mudança no tráfego de e-mails (ex.: nova campanha de marketing, novo cliente
    de e-mail): os sinais de spam (free, remove, money, !, $ e maiúsculas) ficam x fator em
    TODOS os e-mails, legítimos ou não.

    O resultado continua válido no domínio: frequências limitadas a 100%, contagens de
    maiúsculas inteiras e >= 1, e média <= maior sequência <= total.
    """
    saida = df.copy()
    for coluna in config.COLUNAS_SINAL_SPAM:
        if coluna in config.COLUNAS_FREQUENCIA:
            saida[coluna] = (saida[coluna] * fator).clip(upper=100.0)
    total = np.maximum(1, np.round(saida["capital_run_length_total"] * fator)).astype(int)
    maior = np.maximum(1, np.round(saida["capital_run_length_longest"] * fator)).astype(int)
    maior = np.minimum(maior, total)
    media = np.minimum(np.maximum(1.0, saida["capital_run_length_average"] * fator), maior)
    saida["capital_run_length_total"] = total
    saida["capital_run_length_longest"] = maior
    saida["capital_run_length_average"] = media.round(3)
    return saida


def gerar_lote(nome: str, n: int = 800, fator_drift: float = 1.0, semente: int = 1) -> str:
    """Cria data/raw/lotes/<nome>.csv a partir da reserva, opcionalmente com drift."""
    _, _, reserva = dividir_dataset()
    lote = reserva.sample(n=min(n, len(reserva)), random_state=semente).reset_index(drop=True)
    if fator_drift != 1.0:
        lote = aplicar_drift(lote, fator_drift)
    config.DADOS_LOTES.mkdir(parents=True, exist_ok=True)
    caminho = config.DADOS_LOTES / f"{nome}.csv"
    lote.to_csv(caminho, index=False, lineterminator="\n")
    return str(caminho)


def validar_dataframe(df: pd.DataFrame) -> list[str]:
    """Devolve a lista de problemas encontrados (lista vazia = dados válidos).

    Regras do domínio (Spambase):
    - esquema: as 57 colunas + alvo, na ordem, todas numéricas;
    - sem nulos; alvo em {0, 1};
    - frequências (word_freq_*, char_freq_*) são percentuais: entre 0 e 100;
    - maiúsculas: as três medidas >= 1; maior sequência e total são contagens inteiras;
    - coerência: média <= maior sequência <= total de maiúsculas.
    """
    problemas = []
    esperadas = [*config.FEATURES, config.ALVO]
    if list(df.columns) != esperadas:
        faltando = sorted(set(esperadas) - set(df.columns))
        extras = sorted(set(df.columns) - set(esperadas))
        problemas.append(f"esquema diferente do esperado (faltando={faltando}, extras={extras})")
        return problemas
    nao_numericas = [c for c in esperadas if not pd.api.types.is_numeric_dtype(df[c])]
    if nao_numericas:
        problemas.append(f"colunas não numéricas: {nao_numericas[:3]}")
        return problemas
    nulos = int(df.isna().sum().sum())
    if nulos:
        problemas.append(f"{nulos} valores nulos")
    if not df[config.ALVO].dropna().isin([0, 1]).all():
        problemas.append("alvo com valores fora de {0, 1}")
    freq = df[config.COLUNAS_FREQUENCIA]
    negativos = [c for c in config.FEATURES if (df[c] < 0).any()]
    if negativos:
        problemas.append(f"valores negativos em {negativos[:3]}")
    acima_de_100 = [c for c in config.COLUNAS_FREQUENCIA if (freq[c] > 100).any()]
    if acima_de_100:
        problemas.append(f"frequência acima de 100% em {acima_de_100[:3]}")
    media, maior, total = (df[c] for c in config.COLUNAS_MAIUSCULAS)
    if ((df[config.COLUNAS_MAIUSCULAS] < 1).any()).any():
        problemas.append("medidas de maiúsculas abaixo de 1")
    if ((maior.dropna() % 1 != 0).any()) or ((total.dropna() % 1 != 0).any()):
        problemas.append("contagens de maiúsculas (longest/total) não inteiras")
    if (media > maior + 1e-6).any() or (maior > total).any():
        problemas.append("maiúsculas incoerentes (exige média <= maior sequência <= total)")
    return problemas


def listar_lotes() -> list:
    """Arquivos CSV de lotes novos, em ordem alfabética."""
    if not config.DADOS_LOTES.exists():
        return []
    return sorted(config.DADOS_LOTES.glob("*.csv"))


def preparar() -> None:
    """Junta a base com os lotes. Metade de cada lote treina, a outra metade avalia.

    O holdout fixo e as metades de avaliação dos lotes formam a base de avaliação. Assim campeão
    e candidato são comparados no MESMO conjunto, e o campeão nunca viu esses dados.
    """
    treino = [pd.read_csv(config.BASE_TREINO)]
    avaliacao = [pd.read_csv(config.BASE_HOLDOUT)]
    for caminho in listar_lotes():
        lote = pd.read_csv(caminho)
        metade_treino, metade_aval = train_test_split(
            lote, test_size=0.5, stratify=lote[config.ALVO], random_state=SEMENTE_DADOS
        )
        treino.append(metade_treino)
        avaliacao.append(metade_aval)
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    pd.concat(treino, ignore_index=True).to_csv(
        config.TREINO_PROCESSADO, index=False, lineterminator="\n"
    )
    pd.concat(avaliacao, ignore_index=True).to_csv(
        config.AVALIACAO_PROCESSADA, index=False, lineterminator="\n"
    )
    print(f"lotes usados: {len(listar_lotes())}")
    n_treino = sum(len(t) for t in treino)
    n_avaliacao = sum(len(a) for a in avaliacao)
    print(f"treino: {n_treino} linhas | avaliação: {n_avaliacao} linhas")


def validar_tudo() -> int:
    """Valida todos os CSVs de data/raw/. Devolve o código de saída (0 = tudo certo)."""
    arquivos = [config.BASE_TREINO, config.BASE_HOLDOUT, *listar_lotes()]
    codigo = 0
    for caminho in arquivos:
        problemas = validar_dataframe(pd.read_csv(caminho))
        status = "OK" if not problemas else "FALHOU: " + "; ".join(problemas)
        print(f"{caminho.name}: {status}")
        codigo |= 1 if problemas else 0
    return codigo


if __name__ == "__main__":
    comando = sys.argv[1] if len(sys.argv) > 1 else ""
    if comando == "base":
        criar_base()
    elif comando == "preparar":
        preparar()
    elif comando == "validar":
        sys.exit(validar_tudo())
    else:
        print(__doc__)
        sys.exit(2)
