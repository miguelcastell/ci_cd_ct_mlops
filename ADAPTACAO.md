# Relatório de adaptação: filtro de spam com CI/CD/CT

## 1. Problema e dados

- **O que o modelo faz:** classifica um e-mail como **spam** ou **legítimo** a partir de 57 atributos numéricos.
  Quem usaria: um provedor de e-mail ou o servidor de e-mail de uma empresa, para decidir se a mensagem vai para a
  caixa de entrada ou para a pasta de spam.
- **Fonte:** [Spambase, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase)
  (Hopkins, Reeber, Forman e Suermondt, HP Labs, 1999). DOI 10.24432/C53G6X.
- **Licença:** Creative Commons Attribution 4.0 (CC BY 4.0). Não há dados pessoais: só frequências agregadas de
  palavras e caracteres, sem texto, remetente ou destinatário.
- **Tamanho:** 4601 linhas, 57 atributos numéricos (48 `word_freq_*` e 6 `char_freq_*`, em % de 0 a 100, e 3
  `capital_run_length_*`, inteiros ou médias >= 1) e o alvo `spam` (39,4% de spam). Removemos as 6 linhas com atributos
  iguais e rótulos diferentes e as duplicatas: sobram **4204**. Divisão estratificada e fixa (semente 7): **2000**
  para `base_treino.csv`, **1000** para `base_holdout.csv` e **1204** de reserva para gerar lotes novos (sem sobreposição).

## 2. Contrato de qualidade

- **Classe positiva:** spam (`1`).
- **Erro mais caro: falso positivo.** Um e-mail legítimo na pasta de spam (boleto, entrevista, aviso do banco)
  normalmente não é lido, e o prejuízo pode ser grande. Um falso negativo (spam na caixa de entrada) só incomoda e
  o usuário apaga.
- **Métrica principal: precisão do spam >= 0,93.** No máximo 7 de cada 100 mensagens movidas para o spam podem ser
  legítimas. Na prática, filtros comerciais miram precisão ainda maior, mas 0,93 é exigente para uma regressão
  logística nestes dados (o campeão inicial mede 0,959 no holdout) e deixa margem para variação entre lotes.
- **Restrição: recall do spam >= 0,85.** Sem esse piso, um filtro "medroso", que quase nunca marca spam, teria
  precisão alta e não serviria para nada. Aceitamos deixar passar até 15% do spam em troca de proteger os legítimos.
- **Contra o campeão:** o candidato não pode ter precisão menor que a do campeão no mesmo conjunto de avaliação.
- O `params.yaml` usa `class_weight: {0: 2, 1: 1}`: errar um e-mail legítimo pesa o dobro no treino.

## 3. O que mudou, arquivo por arquivo

| Etapa | Arquivo | Mudança |
|---|---|---|
| 1 | `data/raw/*.csv`, `data/fonte/spambase.data` | base e holdout do Spambase (2000 + 1000, sem sobreposição); arquivo original da UCI versionado como origem; `lotes/` vazia |
| 2 | `data.py` | sai `load_breast_cancer`; `carregar_dataset_completo` lê o arquivo da UCI, remove duplicatas e rótulos conflitantes; divisão com semente fixa; CSV sempre com LF |
| 3 | `config.py` | 57 `FEATURES` com nomes legíveis (`char_freq_;` virou `char_freq_semicolon` etc.), `ALVO = "spam"`, `CLASSE_SPAM = 1`, `COLUNAS_SINAL_SPAM` (antes `COLUNAS_TAMANHO`), `COLUNAS_SINAL_LEGITIMO`, novos limiares |
| 4 | `data.py` (`validar_dataframe`) | tipos numéricos; frequências entre 0 e 100%; maiúsculas >= 1; `longest` e `total` inteiros; média <= maior sequência <= total |
| 5 | `features.py`, `params.yaml` | pipeline `log1p` + padronização + regressão logística (`C: 0.1`, `class_weight: {0: 2, 1: 1}`) |
| 6 | `evaluate.py`, `gate.py` | métricas `precisao_spam` e `recall_spam` com `pos_label=1`; o portão compara a **precisão** com a do campeão |
| 7 | `data.py` (`aplicar_drift`), `monitor.py`, `config.py` | drift = sinais de spam (`free`, `remove`, `money`, `!`, `$`, maiúsculas) x fator em todos os e-mails, sem sair do domínio válido; `MIN_FEATURES_DRIFT = 4` (de 57) |
| 8 | `api.py`, `demo/smoke_test.py` | schema `Email` com 57 valores >= 0; resposta `classe` (`spam`/`legitimo`) e `probabilidade_spam`; novo `GET /atributos` |
| 9 | `tests/` | os quatro arquivos adaptados e testes novos por camada (abaixo) |
| 10 | `demo/cenarios.py`, `demo/gerar_lote.py` | cenários 1 a 5 com as colunas do Spambase; lote padrão de 800 e-mails |
| 11 | `models/champion/` | campeão inicial: precisão 0,959, recall 0,890, acurácia 0,941 no holdout |
| 12 | `README.md`, `.github/PULL_REQUEST_TEMPLATE.md` | descrição do novo problema; execução local sem Docker |

**Testes novos:** código: métricas sem erro quando o filtro nunca diz spam; drift simulado mantém os dados válidos;
drift só mexe nos sinais de spam. Dados: frequência acima de 100%; maior sequência de maiúsculas maior que o total;
contagem de maiúsculas não inteira. Modelo: direcional (mais `george`/`hp`/`hpl`, palavras do dono da caixa, não
aumentam a chance de spam); taxa de falso positivo nos legítimos <= 5%. API: frequência negativa é rejeitada (422);
`/atributos` devolve a ordem; e-mail sintético típico de spam é classificado como spam. O teste direcional original
virou "mais sinais de spam não reduzem a probabilidade de spam". O teste de memorizar um lote pequeno usa `C = 100`,
porque com 57 atributos e 40 linhas a regularização do `params.yaml` impede a memorização de propósito.

## 4. Evidências (GitHub Actions)

| | Evidência | Execução |
|---|---|---|
| E0 | CI verde do projeto original, antes de qualquer mudança | [CI #42308607702](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42308607702) |
| E1 | CI verde na `main` após a adaptação (3 jobs) | [CI #42308755831](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42308755831) |
| E2 | CI vermelho no step **Testes de código** (cenário 1, [PR #1](https://github.com/miguelcastell/ci_cd_ct_mlops/pull/1)) | [CI #42308913340](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42308913340) |
| E3 | CT com drift (5 de 57 atributos) e portão aprovando (cenário 4, [PR #2](https://github.com/miguelcastell/ci_cd_ct_mlops/pull/2)); após o merge, o CT da `main` abriu o [PR #3](https://github.com/miguelcastell/ci_cd_ct_mlops/pull/3) com o novo campeão (precisão 0,951 x 0,945) | [CT no PR #42308947964](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42308947964) · [CT na main #42309160728](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42309160728) |
| E4 | CT vermelho no **Portão de qualidade** (cenário 5, [PR #4](https://github.com/miguelcastell/ci_cd_ct_mlops/pull/4)): recall 0,991, precisão 0,628; campeão não mudou | [CT #42309599746](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42309599746) · [CI #42309599684](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42309599684) |
| E5 | CD verde com a tag `v1.0.0`: CI, imagem, teste de fumaça e publicação | [CD #42309818980](https://github.com/miguelcastell/ci_cd_ct_mlops/actions/runs/42309818980) |

Pacote (público): [ghcr.io/miguelcastell/ci_cd_ct_mlops](https://github.com/miguelcastell/ci_cd_ct_mlops/pkgs/container/ci_cd_ct_mlops), tags `1.0.0`
e `latest`.

## 5. Limites e riscos

O portão só olha números agregados em um conjunto de avaliação. Ele não garante um bom modelo porque:
(1) o conjunto pode não representar o tráfego real (o Spambase é de 1999 e de uma única caixa de e-mail, a de um
funcionário da HP; `george` e `hp` são sinais de "legítimo" só para ele); (2) os lotes de drift são simulados, e o drift
real costuma ser adversarial (o spammer muda o texto para enganar o filtro); (3) com ~400 e-mails previstos como spam,
uma diferença de 0,005 na precisão é ruído, e a regra "não piorar o campeão" pode aprovar ou reprovar por acaso;
(4) o portão não mede calibração, justiça entre tipos de remetente, latência nem robustez a entradas fora do domínio.

Em produção monitoraríamos: a taxa de e-mails marcados como spam por dia; as reclamações "não é spam" (falso positivo
real) e "é spam" (falso negativo real) dos usuários, que servem de rótulo atrasado; o drift por atributo (KS) e da
distribuição de `probabilidade_spam`; e latência e erros da API.

**Promoção automática (desafio opcional).** Vantagem: o modelo se adapta ao drift em minutos, sem esperar uma pessoa.
Risco: um lote ruim ou envenenado (spammers marcando spam como legítimo) chega à produção sem revisão, e um falso
positivo em massa afeta todos os usuários. Mitigações: lotes rotulados por fonte confiável, canary e rollback pela tag.

## 6. Contribuição e uso de IA

| Integrante | Contribuição |
|---|---|
| Miguel Mantoan Castellani | _a preencher_ |
| _nome_ | _a preencher_ |

**Uso de IA:** usamos o Claude Code (Anthropic) para adaptar o código, os testes e os cenários ao Spambase, calibrar o
drift e os limiares com experimentos locais e escrever a primeira versão deste relatório e do README. A equipe revisou
as decisões (classe positiva, custo do erro, limiares) e as evidências.
