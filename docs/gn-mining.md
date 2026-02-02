# GN-Mining (BM25) commands

Firstly, create `logs/bm25` directory.

Then, run corresponding command to get mined concepts.

- `python -m src.mining.bm25 eval \
  --entities data/mm-go/meta/onto_for_bm25.jsonl \
  --docs_gold data/mm-go/mld/all_concept_for_bm25.jsonl \
  --fields name \
  --k 1 5 10 100 \
  --topk 100 \
  --metrics_out logs/bm25/mm-go_all_concept_metric.json \
  --preds_out logs/bm25/mm-go_mld_all_concept.jsonl \
  --pretty`:
  Mine top-100 BM25 candidates for GO concepts and write metrics plus per-concept candidate lists.


- `python -m src.mining.bm25 eval \
  --entities data/mm-hpo/meta/onto_for_bm25.jsonl \
  --docs_gold data/mm-hpo/mld/all_concept_for_bm25.jsonl \
  --fields name \
  --k 1 5 10 100 \
  --topk 100 \
  --metrics_out logs/bm25/mm-hpo_all_concept_metric.json \
  --preds_out logs/bm25/mm-hpo_mld_all_concept.jsonl \
  --pretty`:
  Mine top-100 BM25 candidates for HPO concepts and write metrics plus per-concept candidate lists.


## Results

### On Gold concepts from MM-HPO

```json
{
  "num_docs": 1002,
  "topk": 100,
  "hit@k": {
    "1": 0.9161676646706587,
    "5": 0.9970059880239521,
    "10": 0.999001996007984,
    "100": 1.0
  },
  "recall@k": {
    "1": 0.9161676646706587,
    "5": 0.9970059880239521,
    "10": 0.999001996007984,
    "100": 1.0
  },
  "MRR": 0.9528212805158915,
  "MRR@10": 0.9527445109780441
}
```

### On Gold concepts from MM-GO

```json
{
  "num_docs": 1238,
  "topk": 100,
  "hit@k": {
    "1": 1.0,
    "5": 1.0,
    "10": 1.0,
    "100": 1.0
  },
  "recall@k": {
    "1": 1.0,
    "5": 1.0,
    "10": 1.0,
    "100": 1.0
  },
  "MRR": 1.0,
  "MRR@10": 1.0
}
```

## Get Gold-derived Negatives for the second-stage cross-encoder training

- `python -m src.datasets.cross_encoder --dataset go build-gold-bm25 --bm25-jsonl logs/bm25/mm-go_mld_all_concept.jsonl --pool-sizes 5 10 15 20 25 30 35 40 45 50 --out-dir data/mm-go/mld/cross-encoder-input --tag gold_bm25`
- `python -m src.datasets.cross_encoder --dataset hpo build-gold-bm25 --bm25-jsonl logs/bm25/mm-hpo_mld_all_concept.jsonl --pool-sizes 5 10 15 20 25 30 35 40 45 50 --out-dir data/mm-hpo/mld/cross-encoder-input --tag gold_bm25`


The training data is placed in `data/mm-go/mld/cross-encoder-input` and `data/mm-hpo/mld/cross-encoder-input`
