PYTHON ?= python
# Venv 3.12 com litert-torch: o do treino (3.14) não tem a pilha TFLite.
PYTHON_TFLITE ?= .venv-tflite/bin/python
DATA_CONFIG ?= configs/data_pipeline.yaml
TRAIN_CONFIG ?= configs/distillation.yaml
TFLITE ?= research/experiments/model_int8.tflite
MOBILE_MODEL ?= mobile/assets/model_int8.tflite

.PHONY: calibrate ingest train loso aggregate prune prune-loso decisao figuras maos quantize eval export test

# Deslocamento profundidade → cor por sessão; a ingestão lê o resultado.
calibrate:
	$(PYTHON) -m research.ingestion.registration --config $(DATA_CONFIG)

ingest:
	$(PYTHON) -m research.ingestion.build_dataset --config $(DATA_CONFIG)

# make train STAGE=teacher FOLD=0; a destilação também pede TRAIN_ARGS="--teacher-run <dir>".
STAGE ?= student_kd
FOLD ?= 0
train:
	$(PYTHON) -m research.train --config $(TRAIN_CONFIG) --stage $(STAGE) --fold $(FOLD) $(TRAIN_ARGS)

# Os 5 folds de um estágio (sem destilação, que pede um professor por fold), e a agregação.
loso:
	for fold in 0 1 2 3 4; do \
		$(PYTHON) -m research.train --config $(TRAIN_CONFIG) --stage $(STAGE) --fold $$fold $(TRAIN_ARGS) || exit 1; \
	done

aggregate:
	$(PYTHON) -m research.training.aggregate --runs-dir research/experiments/runs $(AGG_ARGS)

# make prune RUN=research/experiments/runs/<run do aluno> (o resultado vai para paths.pruned_dir)
prune:
	$(PYTHON) -m research.optimization.pruner --config $(TRAIN_CONFIG) --run $(RUN) $(PRUNE_ARGS)

# L1 e LAMP a 50% nos 5 folds do student_kd, McNemar e T depois da poda (poda_loso_ft0/resumo.json).
prune-loso:
	$(PYTHON) -m research.optimization.prune_loso --config $(TRAIN_CONFIG)

# Cobertura, acerto, top-k e recall por CM do modelo podado (poda_loso_ft0/decisao_61.json).
decisao:
	$(PYTHON) -m research.optimization.calibration --decisao research/experiments/aggregate/poda_loso_ft0

# Atlas das 61 CMs (C4) e matriz de confusão do student_kd, para o capítulo.
figuras:
	$(PYTHON) -m research.figures atlas --out figuras/atlas.jpg
	$(PYTHON) -m research.figures confusao --out figuras/confusao.pdf

# Mãos do catálogo do app: os 21 pontos e a postura de cada CM (Dart) e o sprite sheet
# 3D sintético; o render pede o MediaPipe (z dos pontos) e usa a GPU se houver.
maos:
	$(PYTHON) -m research.data.canonical_hands --config $(TRAIN_CONFIG)
	$(PYTHON) -m research.data.hand_sprites --config $(TRAIN_CONFIG) --out mobile/assets/hands_61.webp

# make quantize RUN=research/experiments/runs/<run do aluno> [QUANT_ARGS="--landmarks float"]
quantize:
	$(PYTHON_TFLITE) -m research.optimization.quantizer --config $(TRAIN_CONFIG) --run $(RUN) $(QUANT_ARGS)

eval:
	$(PYTHON_TFLITE) -m research.optimization.eval_tflite --config $(TRAIN_CONFIG) --run $(RUN)

# A cópia para o Flutter só acontece depois da validação Int8.
export: eval
	cp $(TFLITE) $(MOBILE_MODEL)

test:
	$(PYTHON) -m pytest
