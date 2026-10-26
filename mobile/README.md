# SignEdge mobile

Reconhece as 61 configurações de mão (CM) da LIBRAS em tempo real, 100% offline,
com o modelo Int8 do SignEdge. Uma CM não é um sinal: a tela mostra a forma da
mão, não o significado de uma palavra.

## Compilar e instalar (Android, USB)

Requisitos: Flutter 3.47.7 ou mais novo (o `tflite_flutter` 0.12.1 traz o LiteRT
1.4), Android SDK com `adb`, Java 21 e um celular com Android 8 ou mais novo.

```bash
flutter pub get
flutter build apk --release     # build/app/outputs/flutter-apk/app-release.apk
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Ou `flutter run --release` com o celular conectado.

No celular:

1. *Configurações → Sobre o telefone*: toque 7 vezes em "Versão" (ou "Versão do
   MIUI") para liberar as opções do desenvolvedor, e ligue a **depuração USB**.
2. Xiaomi/Redmi (MIUI/HyperOS): ligue também **Instalar via USB** nas opções do
   desenvolvedor (pede conta Mi e, às vezes, chip ativo). A cada instalação
   aparece "Instalar app via USB?": a tela precisa estar **desbloqueada**, e a
   confirmação expira em segundos (`INSTALL_FAILED_USER_RESTRICTED`).
3. `INSTALL_FAILED_VERSION_DOWNGRADE`: já existe uma versão instalada com
   versionCode maior (builds por ABI, por exemplo). Use `adb install -r -d …` ou
   desinstale a anterior.
4. Na primeira abertura, permita a câmera.

## Pipeline de um frame

1. Câmera YUV 640×480, girada pelo `sensorOrientation` (aparelho em retrato).
   O modelo recebe o frame sem espelho também na frontal; só o desenho espelha.
2. MediaPipe Hands, só com `tflite_flutter`, sem plugin nativo: o detector de
   palma roda quando a mão some, e nos outros frames a ROI vem dos pontos do
   frame anterior. Os dois `.tflite` saíram de `data/models/hand_landmarker.task`
   (`hand_detector.tflite`) e do `hand_landmark_lite` oficial.
3. Vista do modelo, a mesma `EvalView` do treino: lado = 1,5 × caixa justa
   (maior extensão × 1,4, piso de 0,2333 × lado menor da foto), centrada no
   meio da extensão dos 21 pontos, bilinear, borda replicada.
4. `model_int8.tflite`: o aluno destilado podado (LAMP 50%, sem ajuste fino),
   com T = 0,48 ajustada depois da poda e dobrada na cabeça, em Int8. Entradas
   `rgb` (1, 224, 224, 3) float32 em [0, 1] e `landmarks` (1, 21, 2) em pixels da
   vista. Saída: 61 logits já calibrados. Backbone e cabeça em Int8; o ramo dos
   pontos fica em float (`--landmarks float`), porque o LiteRT 1.4 do
   tflite_flutter não tem SQRT em Int8.
5. EMA contínua dos logits (0,35 para o frame novo, limpa quando a mão some) →
   softmax nas 61 CMs → histerese: uma CM entra com 2 frames seguidos ≥ 0,45 e sai
   abaixo de 0,33.

## A tela

- **Card**: a CM ativa com a mão canônica dela (dedos tingidos) e a
  postura dos dedos, o top-3 (`CM x (p%) · CM y · CM z`) com as barras, o aviso
  "confusão frequente" nas CMs 25, 27, 32, 38, 42 e 47 (recall abaixo de 50%
  fora do fold) e o slider do limiar.
- **Catálogo (61 CMs)**: a grade com a mão de cada CM; tocar numa CM amplia e
  mostra a postura. Serve para consultar a forma antes de fazê-la diante da
  câmera. Com o catálogo aberto, a câmera e a inferência ficam pausadas.

A mão canônica são os 21 pontos do quadro do atlas de C4 (nas CMs 10, 11 e 25, um
quadro de C1–C5 nomeado em `FRAME_OVERRIDE`: na 10 e na 11 o detector ergue o
indicador sobre o polegar, na 25 a mão de perfil não dá volume; na 26, o quadro do
atlas espelhado, para o polegar ficar à esquerda), em
`lib/src/logic/canonical_hands.dart`. O desenho é `assets/hands_61.webp`, um
render 3D sintético desses pontos (SDF e sphere tracing); o app não embarca
imagens da base. Os dois saem de `make maos` na raiz do repositório. A postura de
cada CM foi revisada à mão no atlas e não passou por revisão linguística.

Nas 1.160 pausas fora do fold do LOSO (modelo podado, sem o C6), o limiar 0,45
responde em 86,3% das pausas com 80,2% de acerto entre as respondidas, e a CM
verdadeira está no top-3 em 92,4% (top-1 75,2%, top-2 88,9%). São números do
Kinect; o honesto é medir no celular.

Em segundo plano o app libera a câmera e a reabre na volta (~0,9 s); abrir, fechar
e trocar a câmera rodam em fila, e uma falha de abertura aparece na tela. Os
modelos ficam carregados: liberá-los também tirava só ~17 MB do Native Heap e a
volta subia para ~12 s, com os delegates da GPU recompilando (Redmi Note 13 Pro 5G,
debug). Parado, o app ainda pode ser encerrado pelo LMK do MIUI.

## Benchmark isolado do classificador

`lib/bench_main.dart` mede o `invoke` com o mesmo `tflite_flutter` e o mesmo
`XNNPackDelegate` do app, sem câmera nem MediaPipe (mediana de 5 blocos de 200
após 30 de aquecimento, modelos alternados na mesma sessão):

```bash
flutter build apk --release -t lib/bench_main.dart
adb install -r -d build/app/outputs/flutter-apk/app-release.apk
adb push <modelo>.tflite /storage/emulated/0/Android/data/br.ufpr.signedge.signedge_mobile/files/bench/
adb logcat -s flutter:I | grep BENCH
```

No Redmi Note 10S: 8,6 ms (podado) contra 8,8 ms (denso) em Int8 com os pontos
em float, e 17,1 ms em FP32; entre sessões, o Int8 varia de 8,6 a 13,0 ms
conforme o núcleo que o sistema escolhe. Reinstale o app normal depois.

## Prática do minicurso

1. No Colab, rode `notebooks/minicurso_hands_on.ipynb` com o `minicurso.zip`:
   poda, temperatura depois da poda, decisão nas 61 classes, Int8 e o download
   do `lamp50_int8_lmfloat.tflite`.
2. Copie o arquivo para `mobile/assets/model_int8.tflite`, compile e instale
   (seção acima).
3. Abra o catálogo, escolha uma CM, faça a forma diante da câmera e compare o
   card com a última célula do notebook: o top-3 deve trazer a mesma CM.
4. Ponha o celular em modo avião: o app continua funcionando, porque tudo roda
   no aparelho.

## Testes

`flutter test` cobre a decisão (limiar, top-3, histerese), as mãos
canônicas e o catálogo, e compara a geometria, as âncoras, a decodificação da
palma e a ROI com valores gerados no Python (`torch.grid_sample` do treino e a
referência validada contra o MediaPipe oficial, erro mediano de 1,3 px).
