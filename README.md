# PC-GNN no S-FFSD

Aplicação do **PC-GNN** (*Pick and Choose Graph Neural Network*, Liu et al., WWW 2021) à detecção de fraude em transações de cartão de crédito, usando o conjunto **S-FFSD**, que **não foi usado no artigo original**. Trabalho da disciplina de Detecção de Intrusão, continuação de uma replicação do artigo no YelpChi e na Amazon.

O notebook [`executado.ipynb`](executado.ipynb) está salvo **com todas as saídas**, então dá para ler o experimento completo, com tabelas e figura, sem executar nada.

## Perguntas do experimento

1. O PC-GNN, com a configuração dos autores, funciona no S-FFSD? (linha de base, 10 sementes)
2. Ajustar hiperparâmetros e limiar melhora o resultado, e a melhora se sustenta em sementes novas? (busca usando só a validação, depois 10 sementes no teste)

## Resultados

S-FFSD: 77.881 transações, das quais 29.643 rotuladas (17,73 % de fraude). Divisão estratificada dos rotulados: 11.857 treino, 5.869 validação e 11.917 teste. Média ± desvio padrão em 10 sementes, no conjunto de teste:

| Variante | AUC | AP | F1-macro | G-Mean |
|---|---|---|---|---|
| PC-GNN padrão, τ = 0,5 | 0,7633 ± 0,0028 | 0,5940 ± 0,0026 | 0,6598 ± 0,0831 | 0,6491 ± 0,0325 |
| PC-GNN padrão, τ da validação | 0,7633 ± 0,0028 | 0,5940 ± 0,0026 | 0,7232 ± 0,0033 | 0,6113 ± 0,0048 |
| PC-GNN ajustado, τ = 0,5 | 0,8295 ± 0,0037 | 0,6664 ± 0,0049 | 0,6790 ± 0,0961 | 0,6805 ± 0,0412 |
| PC-GNN ajustado, τ da validação | 0,8295 ± 0,0037 | 0,6664 ± 0,0049 | 0,7512 ± 0,0035 | 0,6668 ± 0,0079 |
| HistGradientBoosting (sem grafo) | 0,8618 | 0,7040 | 0,7648 | 0,6839 |
| Regressão logística (sem grafo) | 0,8143 | 0,6393 | 0,7375 | 0,6588 |

O que os números mostram:

- **Hiperparâmetros importam:** o ajuste elevou a AUC de 0,763 para 0,830, em todas as 10 sementes (Wilcoxon pareado, p = 0,002).
- **O limiar importa:** com τ = 0,5, o F1-macro oscila muito entre sementes (desvio de 0,08 a 0,10). Escolher τ na validação reduz o desvio para cerca de 0,003.
- **O grafo não superou os modelos sem grafo:** o `HistGradientBoosting` sobre os mesmos atributos teve AUC maior (0,862) que o PC-GNN ajustado (0,830).

A configuração escolhida pela busca foi `lr=0.001, weight_decay=5e-4, batch_size=128, rho=0.2, alpha=4, keep_ratio=1.0, emb_size=128, feat_norm=none, num_epochs=31`. O teste de Wilcoxon com 10 pares serve só como indicação.

## Como o experimento está organizado

1. Clona o código dos autores ([`Fialhoalves/PC-GNN`](https://github.com/Fialhoalves/PC-GNN)) no commit `9d7d7fa` e cria uma **cópia adaptada**, sem alterar o clone original. A adaptação (+72 / −52 linhas em `src/`) generaliza o agregador de 3 para R relações (o S-FFSD tem 4), aceita o nome `sffsd`, expõe `keep_ratio` e `feat_norm` e grava as probabilidades. Com 3 relações, a saída é idêntica à do original.
2. Baixa o S-FFSD de [`AI4Risk/antifraud`](https://github.com/AI4Risk/antifraud) e confere o SHA-256.
3. Gera os 120 atributos de janela de tempo (121 atributos no total) com uma versão vetorizada da função do `antifraud`, conferida contra uma cópia literal da original.
4. Monta o grafo com 4 relações (Source, Target, Location e Type) e calcula a homofilia de rótulo de cada uma.
5. Treina o PC-GNN padrão (YAML do YelpChi, sem alterações) com 10 sementes.
6. Busca hiperparâmetros **só com a AUC de validação**: 12 configurações sorteadas, as 3 melhores confirmadas com sementes extras e a melhor rodada nas 10 sementes finais. As sementes da busca são diferentes das finais, e o teste não participa de nenhuma escolha.
7. Compara padrão × ajustado, com τ fixo e com τ escolhido na validação, e com dois modelos de referência sem grafo.
8. Exporta tabelas, figura e um manifesto (commit, versões, hashes dos dados, `diff` da adaptação, sementes).

Métricas: AUC, AP (precisão média), F1-macro, G-Mean e F1 de cada classe.

## Como rodar

O notebook foi feito para o **Google Colab** (CPU, plano gratuito com 2 CPUs é suficiente) e usa caminhos como `/content` e o Google Drive.

1. Abra `executado.ipynb` no [Colab](https://colab.research.google.com/) (*Arquivo → Fazer upload de notebook*, ou pelo GitHub).
2. Use o PyTorch que já vem no Colab. **Não** instale o `requirements.txt` do repositório do PC-GNN.
3. Para um teste rápido de ponta a ponta, mude `QUICK_TEST = True` na seção 1 (2 sementes, 2 épocas, 2 configurações).
4. Execute todas as células. O conjunto de dados é baixado automaticamente. Se quiser usar uma cópia sua, coloque `S-FFSD.csv` ou `S-FFSD.zip` em `MyDrive/PC_GNN_data`.
5. Os resultados são gravados em `MyDrive/PC_GNN_sffsd` (ou em `/content/PC_GNN_sffsd` se `USE_DRIVE = False`).

Tempo: a execução completa treina 10 + 10 + 18 modelos em CPU. Estimo algumas horas no Colab gratuito (os tempos de treino por execução estão na seção 13 do notebook). `TUNE_TIME_BUDGET_MIN` limita a duração da busca. As execuções já concluídas são puladas se você reexecutar.

Para rodar fora do Colab, defina a variável de ambiente `WORK_ROOT` e ajuste os caminhos fixos `/content/...` das células que usam `%%writefile`. Isso não foi testado.

### Parâmetros principais (seção 1)

| Variável | Padrão | Efeito |
|---|---|---|
| `SEEDS` | 10 sementes | Sementes finais (linha de base e configuração ajustada) |
| `N_RANDOM_CONFIGS` | 12 | Configurações sorteadas na busca |
| `TOP_K` | 3 | Quantas passam para a confirmação |
| `N_PARALLEL` | 2 | Execuções simultâneas |
| `INCLUDE_ID_FEATURES` | `False` | Inclui `Time` e os códigos de Source/Target/Location/Type como atributos |
| `QUICK_TEST` | `False` | Teste rápido |

### Dependências

Python 3 com `numpy`, `scipy`, `scikit-learn`, `torch` (CPU), `PyYAML`, `pandas` e `matplotlib`, além do `git`. Veja [`requirements.txt`](requirements.txt). A execução registrada no notebook usou `scikit-learn 1.6.1` e `torch 2.14.1+cpu`.

## Reprodutibilidade

- Commit fixo do código dos autores (`9d7d7fae491081178b6e12e193fb89cfc330f2be`).
- SHA-256 do `S-FFSD.csv`: `a2d78b983dfacaae394e4d69ece46ae880fd6bd4761bc5ee759977ba12b185c6`.
- Sementes fixas e divisão de dados fixa (`random_state=2`).
- O treino roda em CPU com execuções em paralelo, então pequenas diferenças numéricas entre ambientes são esperadas.

## Referências

- Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, Q. He. *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection*. The Web Conference (WWW), 2021.
- [`AI4Risk/antifraud`](https://github.com/AI4Risk/antifraud): origem do conjunto S-FFSD e da engenharia de atributos.
- [`Fialhoalves/PC-GNN`](https://github.com/Fialhoalves/PC-GNN): código do PC-GNN usado como base.
