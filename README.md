# SignEdge

Reconhecimento das **61 configurações de mão (CM) da LIBRAS** em tempo real e 100% offline num celular Android. Um professor RGB-D pesado (ConvNeXt-Tiny) é destilado num aluno móvel (MobileNetV4-Conv-Small + ramo dos 21 pontos do MediaPipe), que é calibrado, podado, quantizado em Int8 e embarcado num app Flutter. É o artefato do minicurso *Quantização, Pruning e Implantação de CNNs* (ERCEMAPI).

Uma CM é a forma da mão, **não um sinal**: o app mostra "CM 38", não o significado de uma palavra.

```text
LIBRAS-HC-RGBDS (Kinect, .oni/.skl)  → research/ingestion  → LMDB (7.948 amostras, 1.400 pausas)
                                         configs/data_pipeline.yaml
LMDB → research/train.py (LOSO 4+1)  → teacher RGB-D → student_kd   configs/distillation.yaml
student_kd → research/optimization   → T (calibração) → poda → Int8 PTQ → model_int8.tflite
model_int8.tflite → mobile/ (Flutter) → MediaPipe (palma + 21 pontos) → vista 224 → decisão
```

## Protocolo

- **LOSO 4+1** sobre os atores C1–C5: treina em 4, mede no 5º, nos 5 folds. Passos fixos, último checkpoint (média EMA dos pesos), sem escolher nada pelo ator medido. Hiperparâmetros pela média dos folds.
- **C6**, outra pessoa gravada em 2012 em outro lugar, é o teste final: o modelo treinado em C1–C5 (`--all-actors`) foi medido **uma única vez**, depois de a receita congelar.
- **Unidade = pausa** (média dos logits dos frames de um trecho parado da mão), não o frame.
- A base de 2013 (Porfirio et al.) usou 70/30 aleatório com **o mesmo ator no treino e no teste** e reportou Rank-3: os números não são comparáveis aos daqui.

## Resultados

### LOSO 4+1, 61 classes (5 folds, 1.160 pausas)

| modelo | pausa (média ± dp) | fechada nas 14 da demo |
|---|---|---|
| só os 21 pontos (MLP) | 68,9 ± 3,9% | 96,2% |
| aluno, só CE | 72,7 ± 5,4% | 95,9% |
| professor RGB-D | 75,3 ± 3,4% | 94,4% |
| **aluno destilado (student_kd)** | **75,8 ± 4,8%** | 95,5% |

KD contra CE: +3,1 pts, ganha nos 5 folds, McNemar p = 0,002. A curva por número de sinalizantes de treino (54 → 64 → 69 → 69%) mostra o gargalo: diversidade de pessoas, não augmentation nem capacidade.

### Teste final no C6 (240 pausas, medido uma vez)

| modelo (treinado em C1–C5) | pausa 61 | fechada 14 | ECE |
|---|---|---|---|
| professor RGB-D (referência) | 79,2% | – | 0,066 |
| student_kd FP32, T = 1 | 74,6% | 90,9% | 0,126 |
| student_kd Int8 + 1/T | 75,8% | 92,7% | 0,044 |
| **embarcado** (Int8, ramo dos pontos em float) | **76,2%** | **92,7%** | – |

O LOSO previu o C6 (75,8 ± 4,8% → 74,6–76,2%). T = 0,76 foi ajustado nas predições fora do fold do LOSO, sem olhar o C6.

### Compressão

| | resultado |
|---|---|
| Int8 PTQ | 10,0 → **3,0 MiB**, sem perda de acurácia (top-1 igual ao FP32 em ~96% dos frames) |
| poda 50%, L1 uniforme, sem fine-tuning | −32 pts nos 5 folds (colapsa) |
| **poda 50%, LAMP, sem fine-tuning** | −0,5 pt (McNemar p = 0,55); 1,7 MiB comprimido (−29%) |
| poda e o `.tflite` | mesmo número de bytes e mesma latência: o XNNPACK Int8 não tem kernel esparso |

O app usa o Int8 **denso**: a poda não ganha nada dentro do APK e muda a calibração.

### App (Flutter, Android)

| aparelho | câmera | mão + classificação processadas |
|---|---|---|
| Redmi Note 10S (Helio G95), boa luz | ~25 fps | **22–24 fps** |
| Redmi Note 13 (Snapdragon 685) | 20–25 fps | 11–15 fps |

Com pouca luz o auto-exposure baixa a câmera para ~14 fps, que vira o teto do app. Detalhes do pipeline, do contrato do modelo e do vocabulário do soletrador em [mobile/README.md](mobile/README.md).

## Como rodar

Dois ambientes Python: `.venv` (3.14, treino, PyTorch CUDA) e `.venv-tflite` (3.12, `litert-torch`, conversão e notebooks). O TensorFlow/LiteRT não vai no venv de treino.

```bash
make ingest                                    # .oni/.skl → LMDB
make loso STAGE=student_ce PYTHON=.venv/bin/python
.venv/bin/python -m research.train --config configs/distillation.yaml --stage student_kd --fold 0 --teacher-run <run do professor>
.venv/bin/python -m research.train --config configs/distillation.yaml --stage teacher --all-actors   # modelo final
make aggregate AGG_ARGS="--stage student_kd --compare student_ce"
.venv/bin/python -m research.optimization.calibration                         # T e limiar fora do fold
make quantize RUN=<run> QUANT_ARGS="--temperature 0.76 --landmarks float"   # .venv-tflite
make eval RUN=<run>
make prune PYTHON=.venv/bin/python RUN=<run>                                  # LAMP, sem fine-tuning
make test
```

## Notebooks

- [notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): o LMDB, a profundidade alinhada por sessão, a vista de avaliação contra a augmentation de treino e a sonda de atalho de fundo. Precisa da base local.
- [notebooks/minicurso_hands_on.ipynb](notebooks/minicurso_hands_on.ipynb): o laboratório do minicurso (poda L1 contra LAMP, temperatura, Int8, tamanho, latência e deploy). Roda no Colab em CPU com o kit `minicurso.zip`, gerado por `python -m research.optimization.kit --config configs/distillation.yaml --run <student_kd de um fold>`. O kit contém recortes da LIBRAS-HC-RGBDS: confira a licença da base antes de distribuir.

## Limitações

- ~75% numa pessoa nova nas 61 classes: 5 sinalizantes não bastam, e CMs vizinhas (33/34, 42/43, 44–47) são quase o mesmo objeto em 2D.
- Kinect de laboratório → celular: fundo, iluminação, JPEG e o MediaPipe no aparelho mudam o domínio. O número honesto da demo é o medido no celular, que ainda não tem um conjunto gravado.
- O vocabulário da demo e as variantes do soletrador foram escolhidos olhando os mesmos folds que reportamos: são otimistas.
- Letras do soletrador e exemplos de sinais aguardam revisão de pessoa surda, intérprete ou do dicionário do INES.
