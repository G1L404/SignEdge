PYTHON ?= python
# Venv 3.12 com litert-torch: o do treino (3.14) não tem a pilha TFLite.
PYTHON_TFLITE ?= .venv-tflite/bin/python
DATA_CONFIG ?= configs/data_pipeline.yaml
TRAIN_CONFIG ?= configs/distillation.yaml
TFLITE ?= research/experiments/model_int8.tflite
MOBILE_MODEL ?= mobile/assets/model_int8.tflite

.PHONY: calibrate ingest train loso aggregate prune quantize eval export test

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
