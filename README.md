# Error-aware Negative-enhanced Ranking (ENR) Framework for Biomedical Concept Recognition

This repository contains all scripts of data preprocessing, model definition, model training, model prediction, 
model evaluation, and data post-processing about the Error-aware Negative-enhanced Ranking (ENR) Framework.

## Data preparation
Please refer to `docs/data.md` to prepare MM-GO and MM-HPO datasets for the experiments.

## Gold-derived Negative Mining (GN-Mining)
Corresponding details are provided in `docs/gn-mining.md`.

## Negative-derived Negative Mining (EN-Mining)
Corresponding details are provided in `docs/en-mining.md`.

## Re-ranker Training
Corresponding details are provided in `docs/cross-encoder.md`.

## Inference Notebook
We provide an offline inference pipeline notebook in `enr-recognizer-inference.ipynb`.
The best checkpoints used in our framework have been uploaded to HuggingFace, so you can test them with your own queries.

Models are:
```
Bi-encoder (retrieval) checkpoints:
  GO_BIENCODER = "Samantha633/enr-recognizer-biological-process-retriever"
  HPO_BIENCODER = "Samantha633/enr-recognizer-phenotypic-abnormality-retriever"

Cross-encoder (rerank) checkpoints:
  GO_CROSSENCODER = "Samantha633/enr-recognizer-biological-process-reranker"
  HPO_CROSSENCODER = "Samantha633/enr-recognizer-phenotypic-abnormality-reranker"
```
