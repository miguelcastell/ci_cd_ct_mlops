# Filtro de spam com CI/CD/CT

Projeto de MLOps que testa, retreina e entrega automaticamente um **classificador de spam**. Usa **GitHub Actions**,
**pytest**, **ruff**, **DVC**, **FastAPI** e **GitHub Container Registry**. É uma adaptação do projeto
[wesleyasantos/ci_cd_ct_mlops](https://github.com/wesleyasantos/ci_cd_ct_mlops), que classificava tumores de mama.
O relatório da adaptação está em [ADAPTACAO.md](ADAPTACAO.md).

**Problema.** Um provedor de e-mail precisa decidir, para cada mensagem recebida, se ela vai para a caixa de entrada
ou para a pasta de spam. O modelo é uma regressão logística (com `log1p` e padronização) treinada no
[Spambase](https://archive.ics.uci.edu/dataset/94/spambase) (UCI, licença CC BY 4.0): 4601 e-mails, 57 atributos
numéricos (frequências de palavras e caracteres e medidas de letras maiúsculas). Depois de remover as duplicatas,
sobram 4204 e-mails.

**Classe positiva: spam (`1`).** O erro mais caro é o **falso positivo**: um e-mail legítimo (boleto, entrevista de
emprego, aviso do banco) que vai para o spam e ninguém lê. Um spam que chega à caixa de entrada só incomoda.

## Contrato de qualidade

Definido em `src/mlops_demo/config.py`:

- **precisão** da classe spam **>= 0,93** (métrica principal: no máximo 7% do que vai para o spam é legítimo)
- **recall** da classe spam **>= 0,85** (sem ele, um filtro que quase nunca marca spam teria precisão alta)
- o candidato não pode ter **precisão** menor que a do campeão, medida **no mesmo conjunto**

## O que cada letra faz neste projeto

| Letra | Workflow | Quando dispara | O que faz |
|---|---|---|---|
| **CI** | `.github/workflows/ci.yml` | push na `main` e todo Pull Request | lint, validação dos dados, testes de código, dados, modelo e API |
| **CT** | `.github/workflows/ct.yml` | agenda semanal e execução manual (sempre retreinam); lote novo ou `params.yaml` alterado (retreinam só se houver drift) | detecta drift, retreina com DVC, aplica o portão de qualidade e propõe o novo campeão em um PR (ou, no modo automático, publica direto e cria a tag) |
| **CD** | `.github/workflows/cd.yml` | tag `v*.*.*` | roda o CI, constrói a imagem Docker **no runner do GitHub**, faz o teste de fumaça e publica no `ghcr.io` |

Para rodar na sua máquina **não é preciso Docker**: a API sobe com `uvicorn`. O Docker só é usado pelo CD, dentro do
GitHub Actions.

## Estrutura

```
.github/workflows/     ci.yml, ct.yml, cd.yml
src/mlops_demo/        config, data, features, evaluate, train, gate, monitor, api
tests/                 test_codigo, test_dados, test_modelo, test_api
data/fonte/            spambase.data (arquivo original da UCI, origem da base e dos lotes)
data/raw/              base_treino.csv (2000), base_holdout.csv (1000) e lotes/ (lotes novos chegam aqui)
models/champion/       o modelo em produção (versionado no Git)
params.yaml            hiperparâmetros (mudou aqui? o CI treina de novo e confere os limiares)
dvc.yaml               pipeline: preparar e treinar
Dockerfile             imagem da API com o modelo campeão (usada pelo CD)
demo/                  cenários de prática, gerador de lotes, teste de fumaça e execução local
```

---

# Passo a passo

## Parte 1. Rodar na sua máquina

Pré-requisitos: Python 3.10 ou superior e Git.

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash no Windows
# PowerShell:  .venv\Scripts\Activate.ps1
# Linux e macOS: source .venv/bin/activate

pip install -e ".[dev,ct]"

python demo/local.py ci              # o mesmo que o CI: todos os passos devem ficar [ok]
python demo/local.py ct              # drift, retreino com DVC e portão (sem abrir PR)

uvicorn mlops_demo.api:app --reload  # sobe a API em http://localhost:8000 (docs em /docs)
python demo/smoke_test.py http://localhost:8000   # em outro terminal
```

> O `dvc repro` usa o `python` do PATH. Se o ambiente virtual não estiver ativo, ele falha com
> `No module named 'mlops_demo'`.

**API.** `GET /health`, `GET /atributos` (a ordem dos 57 atributos) e `POST /predict`:

```json
// entrada: {"atributos": [57 números >= 0, na ordem de GET /atributos]}
// saída (exemplo): {"classe": "spam", "probabilidade_spam": 0.97, "versao_codigo": "0.1.0"}
```

## Parte 2. Configurar o GitHub (uma vez)

1. Repositório **público**, criado do zero (sem fork).
2. **Settings > Actions > General > Workflow permissions**: marque **Read and write permissions** e
   **Allow GitHub Actions to create and approve pull requests**.
3. **Settings > Secrets and variables > Actions > Variables**: crie `PROMOCAO_AUTOMATICA` com o valor `false`
   (o CT abre um PR com o novo campeão e uma pessoa decide). Com `true`, o CT publica na `main`, cria a tag e dispara o
   CD sozinho; nesse caso a `main` não pode ter proteção que exija PR.

## Parte 3. Cenários (as evidências)

Cada cenário muda uma coisa no projeto. Rode sempre em uma branch nova e abra um Pull Request:

```bash
git switch main && git pull
git switch -c cenario-N
python demo/cenarios.py aplicar N
git add -A && git commit -m "cenario N" && git push -u origin cenario-N
```

| N | O que muda | O que se observa |
|---|---|---|
| 1 | `separar_xy` perde a última coluna (`capital_run_length_total`) | CI vermelho em **Testes de código** |
| 2 | nulos em `word_freq_free`, `char_freq_dollar` negativo e `word_freq_money` = 250% | CI vermelho em **Validar os dados brutos** |
| 3 | regularização extrema (`C: 0.00001`): o filtro nunca marca spam | CI vermelho em **Testes de modelo** (precisão e recall 0) |
| 4 | lote de 800 e-mails com drift: `free`, `remove`, `money`, `!`, `$` e maiúsculas x2 em **todos** os e-mails | CI verde. CT: drift em 5 de 57 atributos, o campeão cai para 0,91 de precisão no lote, o candidato retreinado passa no portão |
| 5 | `class_weight: {0: 1, 1: 50}` | recall ~0,99, precisão ~0,63. CI vermelho e, com o lote do cenário 4 na `main`, o CT fica vermelho no portão |

Para listar: `python demo/cenarios.py listar`.

**Cenário 4 (E3).** No PR, abra o workflow **CT** e o **Summary**: relatório de drift e tabela do portão. Faça o
merge. O CT roda na `main` e abre o PR **"ct: novo modelo campeão aprovado no portão de qualidade"**. O CI desse PR
aparece como **Action required** (o PR foi aberto pelo `GITHUB_TOKEN`): clique em **Approve and run workflows**,
revise (só `models/champion/` muda) e faça o merge.

**Cenário 5 (E4).** Com o `lote_01.csv` já na `main`, o PR que muda o `params.yaml` dispara o CT, o drift continua
acima do gatilho, o candidato é treinado e o portão reprova pela precisão. Nada é promovido. Feche o PR sem merge.

**Entrega (E5).** Com a `main` verde:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

O **CD** roda o CI, constrói a imagem, sobe o container no runner, faz o teste de fumaça e publica em
`ghcr.io/<usuario>/<repositorio>`. Em **Packages > Package settings > Change visibility**, deixe a imagem **Public**.

## Comandos úteis

| Comando | Para que serve |
|---|---|
| `python -m mlops_demo.data base` | recria `base_treino.csv` e `base_holdout.csv` a partir de `data/fonte/` (semente fixa) |
| `python -m mlops_demo.data validar` | valida esquema e regras do domínio de todos os CSVs de `data/raw/` |
| `python -m mlops_demo.monitor` | compara os lotes novos com a base (teste KS) e diz se o gatilho dispara |
| `dvc repro` | roda `preparar` e `treinar`, refazendo só o que mudou |
| `python -m mlops_demo.gate` | compara candidato e campeão. Sai com código 1 se reprovar |
| `python -m mlops_demo.train --campeao` | treina o campeão inicial na base e grava em `models/champion/` |
| `python demo/gerar_lote.py lote_02 --drift 2.0` | gera um lote novo de e-mails |

## Se algo der errado

| Sintoma | Causa provável | Solução |
|---|---|---|
| CT passa, mas não há PR | permissão da Parte 2 desligada | ligue a permissão e rode o CT de novo (Actions > CT > Run workflow) |
| CI do PR do campeão em "Action required" | PR aberto pelo `GITHUB_TOKEN` | clique em **Approve and run workflows** |
| `dvc repro` falha com "No module named" | ambiente virtual inativo | ative o ambiente e rode de novo |
| `dvc repro` falha com "is git-ignored" | `dvc.lock` no `.gitignore` | não ignore o `dvc.lock` |
| Cenário não aplica ("trecho não encontrado") | a mudança já foi aplicada nesta branch | volte à `main` e crie outra branch |
