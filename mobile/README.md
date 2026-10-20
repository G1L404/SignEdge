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
4. `model_int8.tflite`: `rgb` (1, 224, 224, 3) float32 em [0, 1] e `landmarks`
   (1, 21, 2) em pixels da vista. Saída: 61 logits já calibrados (1/T na cabeça).
   Backbone e cabeça em Int8; o ramo dos pontos fica em float (`--landmarks
   float`), porque o LiteRT 1.4 do tflite_flutter não tem SQRT em Int8.
5. EMA contínua dos logits (0,35 para o frame novo, limpa quando a mão some) →
   softmax → as variantes de cada letra somam num grupo → histerese: uma CM
   entra com 2 frames seguidos ≥ 0,45 e sai abaixo de 0,33. O card distingue
   "em transição" (abaixo do limiar) de "CM fora da demo" (reconhecida com
   confiança, mas fora do soletrador). O soletrador digita depois de 0,75 s.

## Vocabulário: científico (14) e soletrador (17)

As métricas do C6 e dos relatórios usam o vocabulário científico de 14 CMs
(`vocabularies.demo` em configs/distillation.yaml). O app expande o soletrador
para 17 letras (+ O, V, D) com variantes (alofones) agrupadas: F = 44 + 45/46/47,
L = 38 + 15, B = 53 + 56, C = 26 + 22, O = 21 + 20, V = 49 + 50, D = 14 + 16.
Os grupos vieram da confusão fora do fold do LOSO (1160 pausas, sem o C6) e foram
conferidos no atlas. Nessas pausas: fechada 96,2%; recall aberto 88,9% e falso
aceite 5,5% no limiar 0,45 do app (86,3% e 4,4% com 0,55). São números
otimistas (o vocabulário foi escolhido nelas); o honesto é medir no celular.

## Testes

`flutter test` compara a geometria, as âncoras, a decodificação da palma e a
ROI com valores gerados no Python (`torch.grid_sample` do treino e a referência
validada contra o MediaPipe oficial, erro mediano de 1,3 px).

## Revisão pendente

`assets/handshapes.json` só tem letra onde a correspondência foi conferida
(CM1 = A, CM3 = Y, CM38 = L). Os exemplos de sinais esperam revisão de pessoa
surda, intérprete ou do dicionário do INES; depois, `"review": "revisado"`.
