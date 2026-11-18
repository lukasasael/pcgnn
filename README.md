# Replicação do PC-GNN: aprendizado desbalanceado em grafos para detecção de fraudes

Trabalho da disciplina **CIN0114 – Técnicas de Ataque e Detecção de Intrusão** (CIn-UFPE, 2026.2) sobre o artigo *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection* (Liu et al., WWW 2021), o **PC-GNN**.

**Equipe:** Caio Rocha Calado, Lukas Bacelar, Mariana Melo dos Santos e Vinicius Alves Fialho.

O trabalho tem duas partes, cada uma em um notebook:

| Parte | Notebook | O que faz |
|---|---|---|
| 1. Dados do artigo | [`pcgnn_replicacao_artigo.ipynb`](pcgnn_replicacao_artigo.ipynb) | Reproduz o experimento principal do artigo no **YelpChi** (32 e 100 atributos) e na **Amazon** (25 atributos), com 10 sementes cada |
| 2. Conjunto adicional | [`executado.ipynb`](executado.ipynb) | Aplica o mesmo método ao **S-FFSD**, conjunto que o artigo não usa, e ajusta hiperparâmetros e limiar a esse conjunto |

Os dois notebooks estão salvos **com as saídas**, então dá para ler os experimentos completos sem executar nada. O código dos autores é usado como base em [`Fialhoalves/PC-GNN`](https://github.com/Fialhoalves/PC-GNN) (commit `9d7d7fa`), sem alterações na parte 1. Na parte 2, o código foi adaptado apenas para rodar com quatro relações e permitir o ajuste de hiperparâmetros, sem alterar o método. O relatório em PDF descreve as duas partes.

## Parte 1: dados do artigo (YelpChi e Amazon)

O notebook [`pcgnn_replicacao_artigo.ipynb`](pcgnn_replicacao_artigo.ipynb):

1. Clona o código dos autores, sem alterá-lo, e registra o commit usado.
2. Obtém os dados de três experimentos e confere cada um com a Tabela 1 do artigo (SHA-256 dos arquivos e impressão digital do grafo):
   - `yelp`: YelpChi com 32 atributos (versão pública, do CARE-GNN);
   - `yelp100`: YelpChi com 100 atributos (versão antiga, do histórico do DGFraud, commit `9446e34`);
   - `amazon`: Amazon com 25 atributos (CARE-GNN).
3. Gera as listas de adjacência com a função `sparse_to_adjlist` do repositório. Na Amazon esse trecho de `data_process.py` está comentado, então o notebook aplica a mesma função por um script auxiliar.
4. Treina o PC-GNN com 10 sementes por experimento, usando os arquivos de configuração dos autores (51 épocas, como no repositório; o artigo cita 100).
5. Compara média e desvio padrão com a linha PC-GNN das Tabelas 3 e 5 do artigo e, no YelpChi, compara 32 e 100 atributos semente a semente (Wilcoxon pareado).

Os resultados de cada semente são gravados em disco, então uma execução interrompida retoma de onde parou, e resultados de dados diferentes nunca se misturam.

### Resultados (teste, média ± desvio padrão de 10 sementes)

| Experimento | Fonte | F1-macro | AUC | G-Mean |
|---|---|---|---|---|
| YelpChi, 32 atributos | Replicação | 0,6499 ± 0,0490 | 0,8166 ± 0,0034 | 0,6935 ± 0,0476 |
| YelpChi, 100 atributos | Replicação | 0,5384 ± 0,0382 | 0,6932 ± 0,0079 | 0,6281 ± 0,0206 |
| YelpChi | Artigo | 0,6300 ± 0,0230 | 0,7987 ± 0,0014 | 0,7160 ± 0,0130 |
| Amazon, 25 atributos | Replicação | 0,8785 ± 0,0082 | 0,9639 ± 0,0027 | 0,8986 ± 0,0118 |
| Amazon | Artigo | 0,8956 ± 0,0077 | 0,9586 ± 0,0014 | 0,9030 ± 0,0044 |

- A **AUC** foi reproduzida: na Amazon ficou próxima do artigo e, no YelpChi com 32 atributos, ficou acima.
- As métricas que dependem do **limiar** (F1-macro e G-Mean) variam muito entre sementes no YelpChi.
- A versão de **100 atributos** do YelpChi piorou todas as métricas, então ela provavelmente não é a usada pelos autores.
- **Não cobre:** os *baselines* (GCN, GAT, GraphSAINT, CARE-GNN...) e as ablações PC-GNN\P e PC-GNN\C, que não existem no repositório dos autores. O relatório compara com os números que o artigo reporta.

### Como rodar a parte 1

Feita para o **Google Colab** (CPU). Use o PyTorch do Colab e **não** instale o `requirements.txt` do repositório do PC-GNN, que pede versões antigas. Abra o notebook, execute de cima para baixo e, para um teste rápido, mude `QUICK_TEST = True` na seção de controles. Os dados são baixados automaticamente; se preferir, coloque `YelpChi.mat`, `Amazon.mat` e `YelpChi_100d.mat` na pasta do Drive indicada no notebook. Tempo aproximado por semente: 10 a 12 min no YelpChi e 2 a 3 min na Amazon.

## Parte 2: conjunto de dados adicional (S-FFSD)

Aplicação do PC-GNN, sem alterar o método, à detecção de fraude em transações de cartão de crédito com o **S-FFSD**, que **não foi usado no artigo original**. O notebook [`executado.ipynb`](executado.ipynb) tem o experimento completo, com tabelas e figura.

### Perguntas do experimento

1. O PC-GNN, com a configuração dos autores, funciona no S-FFSD? (linha de base, 10 sementes)
2. Ajustar hiperparâmetros e limiar melhora o resultado, e a melhora se sustenta em sementes novas? (busca usando só a validação, depois 10 sementes no teste)

### Resultados

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

A configuração escolhida pela busca foi `lr=0.001, weight_decay=5e-4, batch_size=128, rho=0.2, alpha=4, keep_ratio=1.0, emb_size=128, feat_norm=none, num_epochs=31`. Com `keep_ratio=1.0`, a subamostragem do *Choose* mantém todos os vizinhos. O teste de Wilcoxon com 10 pares serve só como indicação.

### Como o experimento está organizado

1. Clona o código dos autores ([`Fialhoalves/PC-GNN`](https://github.com/Fialhoalves/PC-GNN)) no commit `9d7d7fa` e cria uma **cópia adaptada**, sem alterar o clone original. A adaptação (+72 / −52 linhas em `src/`) não altera o método. Duas mudanças são necessárias para rodar no S-FFSD: generalizar o agregador de 3 para R relações (o S-FFSD tem 4) e aceitar o nome `sffsd`. As outras duas servem só para o ajuste: expor `keep_ratio` e `feat_norm` como opções de configuração (na linha de base, ficam nos valores do original) e gravar as probabilidades, para escolher o limiar na validação. Com 3 relações, a saída é idêntica à do original.
2. Baixa o S-FFSD de [`AI4Risk/antifraud`](https://github.com/AI4Risk/antifraud) e confere o SHA-256.
3. Gera os 120 atributos de janela de tempo (121 atributos no total) com uma versão vetorizada da função do `antifraud`, conferida contra uma cópia literal da original.
4. Monta o grafo com 4 relações (Source, Target, Location e Type) e calcula a homofilia de rótulo de cada uma.
5. Treina o PC-GNN padrão (YAML do YelpChi, sem alterações) com 10 sementes.
6. Busca hiperparâmetros **só com a AUC de validação**: 12 configurações sorteadas, as 3 melhores confirmadas com sementes extras e a melhor rodada nas 10 sementes finais. As sementes da busca são diferentes das finais, e o teste não participa de nenhuma escolha.
7. Compara padrão × ajustado, com τ fixo e com τ escolhido na validação, e com dois modelos de referência sem grafo.
8. Exporta tabelas, figura e um manifesto (commit, versões, hashes dos dados, `diff` da adaptação, sementes).

Métricas: AUC, AP (precisão média), F1-macro, G-Mean e F1 de cada classe.

### Como rodar

O notebook foi feito para o **Google Colab** (CPU, plano gratuito com 2 CPUs é suficiente) e usa caminhos como `/content` e o Google Drive.

1. Abra `executado.ipynb` no [Colab](https://colab.research.google.com/) (*Arquivo → Fazer upload de notebook*, ou pelo GitHub).
2. Use o PyTorch que já vem no Colab. **Não** instale o `requirements.txt` do repositório do PC-GNN.
3. Para um teste rápido de ponta a ponta, mude `QUICK_TEST = True` na seção 1 (2 sementes, 2 épocas, 2 configurações).
4. Execute todas as células. O conjunto de dados é baixado automaticamente. Se quiser usar uma cópia sua, coloque `S-FFSD.csv` ou `S-FFSD.zip` em `MyDrive/PC_GNN_data`.
5. Os resultados são gravados em `MyDrive/PC_GNN_sffsd` (ou em `/content/PC_GNN_sffsd` se `USE_DRIVE = False`).

Tempo: a execução completa treina 10 + 10 + 18 modelos em CPU. Estimo algumas horas no Colab gratuito (os tempos de treino por execução estão na seção 13 do notebook). `TUNE_TIME_BUDGET_MIN` limita a duração da busca. As execuções já concluídas são puladas se você reexecutar.

Para rodar fora do Colab, defina a variável de ambiente `WORK_ROOT` e ajuste os caminhos fixos `/content/...` das células que usam `%%writefile`. Isso não foi testado.

#### Parâmetros principais (seção 1)

| Variável | Padrão | Efeito |
|---|---|---|
| `SEEDS` | 10 sementes | Sementes finais (linha de base e configuração ajustada) |
| `N_RANDOM_CONFIGS` | 12 | Configurações sorteadas na busca |
| `TOP_K` | 3 | Quantas passam para a confirmação |
| `N_PARALLEL` | 2 | Execuções simultâneas |
| `INCLUDE_ID_FEATURES` | `False` | Inclui `Time` e os códigos de Source/Target/Location/Type como atributos |
| `QUICK_TEST` | `False` | Teste rápido |

#### Dependências

Python 3 com `numpy`, `scipy`, `scikit-learn`, `torch` (CPU), `PyYAML`, `pandas` e `matplotlib`, além do `git`. Veja [`requirements.txt`](requirements.txt). A execução registrada no notebook usou `scikit-learn 1.6.1` e `torch 2.14.1+cpu`.

### Reprodutibilidade

- Commit fixo do código dos autores (`9d7d7fae491081178b6e12e193fb89cfc330f2be`).
- SHA-256 do `S-FFSD.csv`: `a2d78b983dfacaae394e4d69ece46ae880fd6bd4761bc5ee759977ba12b185c6`.
- Sementes fixas e divisão de dados fixa (`random_state=2`).
- O treino roda em CPU com execuções em paralelo, então pequenas diferenças numéricas entre ambientes são esperadas.

### Referências do S-FFSD

- Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, Q. He. *Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection*. The Web Conference (WWW), 2021.
- [`AI4Risk/antifraud`](https://github.com/AI4Risk/antifraud): origem do conjunto S-FFSD e da engenharia de atributos.
- [`Fialhoalves/PC-GNN`](https://github.com/Fialhoalves/PC-GNN): código do PC-GNN usado como base.
