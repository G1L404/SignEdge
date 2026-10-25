# SignEdge mobile

Reconhece as configurações de mão (CM) da LIBRAS em tempo real, 100% offline,
com o modelo Int8 do SignEdge. Uma CM não é um sinal: a tela mostra a forma
da mão, não o significado de uma palavra.

## Rodar no Android (USB)

1. Android SDK + `adb`, e depuração USB ligada no celular (Android 8+).
2. `flutter pub get`
3. `flutter run --release`

## Pipeline de um frame

1. Câmera YUV 640×480, girada pelo `sensorOrientation` (aparelho em retrato).
   O modelo recebe o frame sem espelho também na frontal; só o desenho espelha.
2. MediaPipe Hands, só com `tflite_flutter`, sem plugin nativo: o detector de
   palma roda quando a mão some, e nos outros frames a ROI vem dos pontos do
   frame anterior. Os dois `.tflite` saíram de `data/models/hand_landmarker.task`.
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
   abaixo de 0,33. O card mostra a CM ativa e o top-3 (`CM x (p%) · CM y · CM z`).

## As 61 configurações

O app classifica e mostra as 61 CMs, sem vocabulário reduzido nem soletrador.
Nas 1.160 pausas fora do fold do LOSO (modelo podado, T = 0,48, sem o C6), o
limiar 0,45 responde em 86,3% das pausas com 80,2% de acerto entre as
respondidas, e a CM verdadeira está no top-3 em 92,4% das pausas (top-1 75,2%,
top-2 88,9%). As CMs 25, 27, 32, 38, 42 e 47 têm recall abaixo de 50% e ganham o
aviso "confusão frequente" no card. São números do Kinect; o honesto é medir no
celular.

Para 17 CMs o card também diz a letra ou o número que a forma lembra
(`assets/handshapes.json`, de Porfirio et al. 2013 e do atlas UFPR), e as vizinhas
que a rede confunde herdam a leitura do grupo (CM 22 lembra o C da 26). Quando o
top-3 divide a confiança entre letras diferentes (≥ 15% cada), aparece "lembra C ou
O". É uma leitura aproximada: a CM é só a forma da mão, e as letras que dependem de
orientação ou movimento não entram. Aguarda revisão de pessoa surda, intérprete ou
do dicionário do INES.

Em segundo plano o app libera a câmera e os três modelos (interpretadores,
delegates e isolates) e os recarrega na volta: parado com tudo carregado, ele
ocupava ~260 MB e o LMK do MIUI o encerrava.

## Testes

`flutter test` compara a geometria, as âncoras, a decodificação da palma e a
ROI com valores gerados no Python (`torch.grid_sample` do treino e a referência
validada contra o MediaPipe oficial, erro mediano de 1,3 px).
