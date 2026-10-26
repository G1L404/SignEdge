# SignEdge

Reconhecimento das **61 configurações de mão (CM) da LIBRAS** em tempo real e 100% offline num celular Android. Um professor RGB-D pesado (ConvNeXt-Tiny) é destilado num aluno móvel (MobileNetV4-Conv-Small + ramo dos 21 pontos do MediaPipe), que é podado, recalibrado, quantizado em Int8 e embarcado num app Flutter. É o artefato do minicurso *Quantização, Pruning e Implantação de CNNs* (ERCEMAPI).

Uma CM é a forma da mão, **não um sinal**: o app mostra "CM 38", não o significado de uma palavra.

```text
LIBRAS-HC-RGBDS (Kinect, .oni/.skl)  → research/ingestion  → LMDB (7.948 amostras, 1.400 pausas)
                                         configs/data_pipeline.yaml
LMDB → research/train.py (LOSO 4+1)  → teacher RGB-D → student_kd   configs/distillation.yaml
student_kd → research/optimization   → poda LAMP → T depois da poda → Int8 PTQ → model_int8.tflite
model_int8.tflite → mobile/ (Flutter) → MediaPipe (palma + 21 pontos) → vista 224 → 61 CMs + top-3
```

## Protocolo

- **LOSO 4+1** sobre os atores C1–C5: treina em 4, mede no 5º, nos 5 folds. Passos fixos, último checkpoint (média EMA dos pesos), sem escolher nada pelo ator medido. Hiperparâmetros pela média dos folds.
- **C6**, outra pessoa gravada em 2012 em outro lugar, é o teste final: o modelo treinado em C1–C5 (`--all-actors`) foi medido **uma única vez**, depois de a receita congelar.
- **Unidade = pausa** (média dos logits dos frames de um trecho parado da mão), não o frame.
- A base de 2013 (Porfirio et al.) usou 70/30 aleatório com **o mesmo ator no treino e no teste** e reportou Rank-3: os números não são comparáveis aos daqui.

## Resultados

### LOSO 4+1, 61 classes (5 folds, 1.160 pausas, treino em bfloat16)

| modelo | pausa (média ± dp) | quadro | F1-macro |
|---|---|---|---|
| só os 21 pontos (MLP) | 68,9 ± 3,9% | 68,8% | 0,666 |
| aluno, só CE | 72,7 ± 5,4% | 71,3% | 0,707 |
| professor RGB-D | 75,3 ± 3,4% | 74,1% | 0,742 |
| **aluno destilado (student_kd)** | **75,8 ± 4,8%** | 74,6% | 0,742 |

KD contra CE: +3,1 pts, ganha nos 5 folds. A curva por número de sinalizantes de treino (54 → 64 → 69 → 69%) mostra o gargalo: diversidade de pessoas, não augmentation nem capacidade.

### Compressão (o pipeline do app)

Poda LAMP 50% em corte único, sem ajuste fino → T ajustada depois da poda nas 1.160 pausas fora do fold → Int8 PTQ com o ramo dos pontos em float32.

| etapa | resultado | fonte |
|---|---|---|
| poda 50%, L1 uniforme, ft0 | 43,87 ± 9,19% nos 5 folds (−31,9 pts) | `aggregate/poda_loso_ft0/resumo.json` |
| **poda 50%, LAMP, ft0** | 75,18 ± 4,43% contra 75,77 ± 4,50% do denso (−0,6 pt, McNemar p = 0,47) | idem |
| temperatura depois da poda | T = 0,76 (denso) → **0,48** (podado); ECE por pausa 0,425 → 0,066 (0,234 com o T do denso) | idem |
| Int8, pontos em float32 | 10,02 → **3,01 MiB**; gzip 2,43 MiB (denso) → **1,73 MiB** (podado) | `exports/*/eval_ood.json` |
| latência isolada, desktop (i7-13620H, 1 thread) | FP32 4,12 ms → Int8 2,34 ms; podado = denso (2,34 contra 2,35) | `bench/desktop.json` |
| latência isolada, Redmi Note 10S | FP32 17,1 ms → Int8 **8,6 ms**; podado = denso (8,6 contra 8,8) | `bench/redmi_note10s.json` |

Os zeros da poda não reduzem o `.tflite` nem aceleram o XNNPACK Int8 (matrizes densas); o ganho é o download comprimido e o mesmo modelo no laboratório e no celular. A versão Int8 integral (2,95 MiB) não executa no Android: o LiteRT 1.4 não tem `SQRT` em Int8.

### Teste final no C6 (240 pausas, medido uma vez)

| modelo (treinado em C1–C5) | pausa | quadro | ECE (quadro) |
|---|---|---|---|
| professor RGB-D (referência) | 79,2% | – | – |
| aluno destilado FP32 | 74,6% | – | – |
| aluno podado FP32, T = 0,48 | 75,4% | 73,9% | 0,057 |
| aluno podado Int8 integral | 75,0% | 73,3% | 0,055 |
| **app: podado, Int8, pontos em float32** | **76,2%** [70,5; 81,2] | 73,4% | 0,057 |

O C6 não escolheu a poda, a temperatura nem o limiar. A concordância top-1 do arquivo do app com o FP32 é 95,7% dos quadros.

### Decisão do app nas 61 classes

Nas 1.160 pausas fora do fold do modelo podado (T = 0,48), `aggregate/poda_loso_ft0/decisao_61.json`:

| limiar | pausas respondidas | acerto entre as respondidas |
|---|---|---|
| 0,30 | 94,7% | 77,5% |
| **0,45 (app)** | **86,3%** | **80,2%** |
| 0,70 | 70,1% | 86,5% |

A CM certa está no top-1 em 75,2% das pausas, no top-2 em 88,9% e no top-3 em 92,4%. As CMs 25, 27, 32, 38, 42 e 47 têm recall abaixo de 50%.

### App (Flutter, Android)

| aparelho | câmera | mão + classificação processadas |
|---|---|---|
| Redmi Note 10S (Helio G95), boa luz | ~25 fps | 22–24 fps |
| Redmi Note 13 (Snapdragon 685) | 20–25 fps | 11–15 fps |

Taxas medidas com o asset Int8 denso anterior, que tem a mesma latência isolada do podado. Com pouca luz o auto-exposure baixa a câmera para ~14–16 fps, que vira o teto do app. Pipeline, contrato do modelo, catálogo das 61 CMs e instalação em [mobile/README.md](mobile/README.md).

## Como rodar

Dois ambientes Python: `.venv` (3.14, treino, PyTorch CUDA) e `.venv-tflite` (3.12, `litert-torch`, conversão e notebooks). O TensorFlow/LiteRT não vai no venv de treino.

```bash
make ingest                                            # .oni/.skl → LMDB
make loso STAGE=student_ce PYTHON=.venv/bin/python
.venv/bin/python -m research.train --config configs/distillation.yaml --stage student_kd --fold 0 --teacher-run <run do professor>
.venv/bin/python -m research.train --config configs/distillation.yaml --stage student_kd --all-actors --teacher-run <professor --all-actors>
make aggregate AGG_ARGS="--stage student_kd --compare student_ce"
make prune-loso PYTHON=.venv/bin/python               # L1 e LAMP nos 5 folds, McNemar e T depois da poda
make decisao PYTHON=.venv/bin/python                  # cobertura, acerto e top-k do modelo podado
make prune PYTHON=.venv/bin/python RUN=<student_kd --all-actors>          # LAMP 50%, sem fine-tuning
make quantize RUN=<run podado> QUANT_ARGS="--temperature 0.48 --landmarks float"   # .venv-tflite
make export RUN=<run podado> TFLITE=research/experiments/exports/<run podado>/model_int8_lmfloat.tflite
make figuras PYTHON=.venv/bin/python                  # figuras/atlas.jpg e figuras/confusao.pdf
make maos PYTHON=.venv/bin/python                     # mãos e posturas do catálogo do app (Dart + sprites 3D)
make test
```

## Notebooks

- [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): o LMDB, a profundidade alinhada por sessão, a vista de avaliação contra a augmentation de treino e a sonda de atalho de fundo. Precisa da base local.
- [notebooks/minicurso_hands_on.ipynb](notebooks/minicurso_hands_on.ipynb): o laboratório do minicurso, na ordem do app: poda L1 contra LAMP, temperatura depois da poda, decisão nas 61 classes (cobertura, acerto e top-3), Int8 com o ramo dos pontos em float32 e o `lamp50_int8_lmfloat.tflite` para o celular. Roda no Colab em CPU com o kit `minicurso.zip`, gerado por `python -m research.optimization.kit --config configs/distillation.yaml --run <student_kd de um fold>` depois do `make prune-loso`. O kit contém recortes da LIBRAS-HC-RGBDS: confira a licença da base antes de distribuir.

## Limitações

- ~75% numa pessoa nova nas 61 classes: 5 sinalizantes não bastam, e CMs vizinhas (46/47, 42/43, 15/38) são quase o mesmo objeto em 2D.
- Kinect de laboratório → celular: fundo, iluminação, JPEG e o MediaPipe no aparelho mudam o domínio. O número honesto do app é o medido no celular, que ainda não tem um conjunto gravado.
- Com o limiar 0,45, cerca de 1 em cada 5 respostas exibidas está errada nas pausas fora do fold; o top-3 contém a certa em 92,4% delas.
- As posturas do catálogo do app foram revisadas à mão no atlas (61 descrições distintas), sem revisão linguística. As mãos do catálogo são um render 3D sintético dos 21 pontos, com o z do MediaPipe: uma ilustração da forma, não uma medida.
