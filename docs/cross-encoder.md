# Listwise Cross-Encoder

After prepared the input for cross-encoder training (refer to `docs/gn-mining.md` and `docs/en-mining.md`), we can train
a cross-encoder with following commands.

For the model used for inference pipeline exploration, we train the model with 20 gold-derived BM25 negatives, and select the best checkpoint by evaluating on the accuracy@1 given 1 positive concept and 20 gold-derived BM25 negatives per positive pair.
- `python -m src.trainers.cross_encoder --seed 42 --loss_type listwise --dev_eval_mode proxy --k_hard 4 --eval_ks 5 10 20 --train_jsonl data/mm-go/mld/cross-encoder-input/train_gold_bm25_20.jsonl --dev_jsonl data/mm-go/mld/bi-encoder-input/dev_biencoder.jsonl --output_dir logs/mm-go/cross_encoder/ckpt`

For the model used for the exploration of EN's effects, we train the model with corresponding data, and we select the best model according to development-set nDCG@100, which directly measures ranking quality over candidate lists.
Before the training, we need to get the development file (and test file) based on the bi-encoder top-100 predictions.

- `python -m src.datasets.bi_encoder build-cross-rerank --data-name go --data-root data --split dev --pred-jsonl logs/bi_encoder/mm-go/top-100-predictions/dev.jsonl --out-jsonl data/mm-go/mld/cross-encoder-input/dev_from_biencoder_top100.jsonl --keep-topk 100`
- `python -m src.datasets.bi_encoder build-cross-rerank --data-name go --data-root data --split test --pred-jsonl logs/bi_encoder/mm-go/top-100-predictions/test.jsonl --out-jsonl data/mm-go/mld/cross-encoder-input/test_from_biencoder_top100.jsonl --keep-topk 100`

# Training

- `python -m src.trainers.cross_encoder --seed 42 --loss_type listwise --dev_eval_mode rerank --k_hard 4 --train_jsonl data/mm-go/mld/cross-encoder-input/train_gold_bm25_20.jsonl --dev_jsonl data/mm-go/mld/cross-encoder-input/dev_from_biencoder_top100.jsonl --output_dir logs/mm-go/cross_encoder/ckpt-gold_bm25_20`


# Prediction

Before get the predictions, make sure the directory for storing output files (like `logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20`) exists.

- `python -m src.evaluation.cross_encoder_predict predict --model_dir logs/cross_encoder/mm-go/ckpt-gold_bm25_20/best --in_jsonl data/mm-go/mld/cross-encoder-input/dev_from_biencoder_top100.jsonl --out_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/dev_biencoder_top100.jsonl --pred_topk 100 --eval_topk 100 --max_len 512 --batch_size 64`
- `python -m src.evaluation.cross_encoder_predict predict --model_dir logs/cross_encoder/mm-go/ckpt-gold_bm25_20/best --in_jsonl data/mm-go/mld/cross-encoder-input/test_from_biencoder_top100.jsonl --out_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/test_biencoder_top100.jsonl --pred_topk 100 --eval_topk 100 --max_len 512 --batch_size 64`

As the model outputs do not include concept id for each predicted term, we need to add this information to prediction files.

- `python -m src.datasets.cross_encoder --dataset go attach-ids-preds --in-pred-jsonl  logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/dev_biencoder_top100.jsonl --out-pred-jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/dev_biencoder_top100.jsonl --concept-catalog-json data/mm-go/meta/biological_process_concept.json`
- `python -m src.datasets.cross_encoder --dataset go attach-ids-preds --in-pred-jsonl  logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/test_biencoder_top100.jsonl --out-pred-jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/test_biencoder_top100.jsonl --concept-catalog-json data/mm-go/meta/biological_process_concept.json`

We provide the predictions from our best setting in `logs/cross_encoder/mm-go/model_outputs/ckpt-macoir_xrt_be_gold_bm25_40` and `logs/cross_encoder/mm-hpo/model_outputs/ckpt-macoir_xrt_be_gold_bm25_50`.

# Evaluation

Before get the evaluation results, make sure the directory for storing output files (like `data/mm-go/mld/cross-encoder-output/ckpt-gold_bm25_20`) exists.

- `python -m src.evaluation.cross_encoder_predict eval --dev_pred_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/dev_biencoder_top100.jsonl --test_pred_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-gold_bm25_20/test_biencoder_top100.jsonl --gold_json data/mm-go/ori/all_wo_mention.json --split_list_json data/mm-go/ori/split_list.json --topk 100 --out_dir data/mm-go/mld/cross-encoder-output/ckpt-gold_bm25_20 --with_score`

Use the predictions we provided in , you can get the evaluation results by running:

- `python -m src.evaluation.cross_encoder_predict eval --dev_pred_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-macoir_xrt_be_gold_bm25_40/dev_biencoder_top100.jsonl --test_pred_jsonl logs/cross_encoder/mm-go/model_outputs/ckpt-macoir_xrt_be_gold_bm25_40/test_biencoder_top100.jsonl --gold_json data/mm-go/ori/all_wo_mention.json --split_list_json data/mm-go/ori/split_list.json --topk 100 --out_dir data/mm-go/mld/cross-encoder-output/macoir_xrt_be_gold_bm25_50 --with_score`
- `python -m src.evaluation.cross_encoder_predict eval --dev_pred_jsonl logs/cross_encoder/mm-hpo/model_outputs/ckpt-macoir_xrt_be_gold_bm25_50/dev_biencoder_top100.jsonl --test_pred_jsonl logs/cross_encoder/mm-hpo/model_outputs/ckpt-macoir_xrt_be_gold_bm25_50/test_biencoder_top100.jsonl --gold_json data/mm-hpo/ori/all_wo_mention.json --split_list_json data/mm-hpo/ori/split_list.json --topk 100 --out_dir data/mm-hpo/mld/cross-encoder-output/macoir_xrt_be_gold_bm25_50 --with_score`
