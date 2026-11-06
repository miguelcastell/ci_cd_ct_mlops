"""Cenários de prática: cada um aplica uma mudança para você ver o pipeline reagir.

Uso (a partir da raiz do projeto, sempre em uma branch nova):
    python demo/cenarios.py listar
    python demo/cenarios.py aplicar 1

Depois de aplicar: git add -A, git commit, git push e abra o Pull Request.
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from mlops_demo.data import gerar_lote  # noqa: E402

DESCRICAO = {
    1: "Bug no pré-processamento: some o capital_run_length_total. Esperado: CI falha (código).",
    2: "Dados corrompidos (nulos, frequência < 0 e > 100%). Esperado: CI falha em validar dados.",
    3: "C muito pequeno: o filtro para de marcar spam. Esperado: CI falha em test_modelo.",
    4: "Lote novo com drift (sinais de spam x2). Esperado: CI verde, CT retreina e portão aprova.",
    5: "class_weight extremo no spam: recall ~1 e precisão baixa. Esperado: CI e portão reprovam.",
}


def _substituir(arquivo: Path, antigo: str, novo: str) -> None:
    texto = arquivo.read_text(encoding="utf-8")
    if antigo not in texto:
        raise SystemExit(f"Trecho não encontrado em {arquivo.name}. O cenário já foi aplicado?")
    arquivo.write_text(texto.replace(antigo, novo, 1), encoding="utf-8")
    print(f"alterado: {arquivo.relative_to(RAIZ)}")


def cenario_1() -> None:
    _substituir(
        RAIZ / "src" / "mlops_demo" / "features.py",
        "X = df[FEATURES].copy()",
        "X = df[FEATURES[:-1]].copy()",
    )


def cenario_2() -> None:
    caminho = RAIZ / "data" / "raw" / "base_treino.csv"
    df = pd.read_csv(caminho)
    df.loc[0:4, "word_freq_free"] = None
    df.loc[5:7, "char_freq_dollar"] = -1.0
    df.loc[8:9, "word_freq_money"] = 250.0
    df.to_csv(caminho, index=False, lineterminator="\n")
    print(
        "alterado: data/raw/base_treino.csv (5 nulos em word_freq_free, 3 char_freq_dollar"
        " negativos, 2 word_freq_money = 250%)"
    )


def cenario_3() -> None:
    _substituir(RAIZ / "params.yaml", "C: 0.1", "C: 0.00001")


def cenario_4() -> None:
    caminho = gerar_lote("lote_01", n=800, fator_drift=2.0, semente=1)
    print(
        "criado:",
        Path(caminho).relative_to(RAIZ),
        "(free, remove, money, !, $ e maiúsculas x2 em 800 e-mails)",
    )


def cenario_5() -> None:
    _substituir(RAIZ / "params.yaml", "class_weight: {0: 2, 1: 1}", "class_weight: {0: 1, 1: 50}")


CENARIOS = {1: cenario_1, 2: cenario_2, 3: cenario_3, 4: cenario_4, 5: cenario_5}

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "listar":
        for numero, texto in DESCRICAO.items():
            print(f"{numero}. {texto}")
    elif len(sys.argv) == 3 and sys.argv[1] == "aplicar" and sys.argv[2].isdigit():
        numero = int(sys.argv[2])
        if numero not in CENARIOS:
            raise SystemExit("Cenário inexistente. Use: python demo/cenarios.py listar")
        print(f"Cenário {numero}: {DESCRICAO[numero]}")
        CENARIOS[numero]()
    else:
        print(__doc__)
        sys.exit(2)
