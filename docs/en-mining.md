# EN-Mining

## MA-COIR

### Step 1: Index construction

Firstly, we need to build the concept indices utilized by MA-COIR.
Please create a new directory `logs/macoir`.

In our default setting, we use only semantic relations among ontology concepts to build edges, resulting SSI that
published by Liu et. al (2026).

If you prefer OSI (which means ontological relations among concepts are used for edge construction) or OSSI (both
ontological relations and semantic relations), you could replace `--use_semantic_similarity`
with `--use_ontology_structure` or `--use_semantic_similarity --use_ontology_structure`.
Moreover, revise the `--indexing_path` to your prefer name instead of `ssi_id_sid.json`.

And we use SapBERT ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") for semantic embedding calculation.

If you prefer other models, you can set `--model_name` with corresponding Huggingface Model Card.

#### Fresh run (compute embeddings)

- `python -m src.mining.macoir_index --ontology_file data/mm-go/meta/go-basic.json --concept_file data/mm-go/meta/biological_process_concept.json --concept_id_file data/mm-go/meta/biological_process_concept_id_list.json --embedding_pickle logs/macoir/mm-go/embeddings.pkl --graph_pickle logs/macoir/mm-go/graph.pkl --partitioned_graph_pickle logs/macoir/mm-go/graph_partitioned.pkl --checkpoint_dir logs/macoir/mm-go/index_ckpt --csv_path logs/macoir/mm-go/search_id.tsv --indexing_path logs/macoir/mm-go/ssi_id_sid.json --use_semantic_similarity --L 10 --M 10 --target_leaf_size 10 --save_step 1`

Build MA-COIR index for GO (encode concepts, build graph, partition, export mappings).

Args:

--ontology_file (raw ontology),
--concept_file (name/synonym JSON),
--concept_id_file (allowed IDs),

--embedding_pickle/--graph_pickle/--partitioned_graph_pickle (outputs),
--checkpoint_dir (partition checkpoints),

--csv_path/--indexing_path (exported mappings),
--use_semantic_similarity/--use_ontology_structure (edge sources),

--L/--M (FAISS neighbors/kept neighbors),
--target_leaf_size (cluster size),
--save_step (checkpoint frequency).

- `python -m src.mining.macoir_index --ontology_file data/mm-hpo/meta/hp.json --concept_file data/mm-hpo/meta/phenotypic_abnormality_concept.json --concept_id_file data/mm-hpo/meta/phenotypic_abnormality_concept_id_list.json --embedding_pickle logs/macoir/mm-hpo/embeddings.pkl --graph_pickle logs/macoir/mm-hpo/graph.pkl --partitioned_graph_pickle logs/macoir/mm-hpo/graph_partitioned.pkl --checkpoint_dir logs/macoir/mm-hpo/index_ckpt --csv_path logs/macoir/mm-hpo/search_id.tsv --indexing_path logs/macoir/mm-hpo/ssi_id_sid.json --use_semantic_similarity --L 10 --M 10 --target_leaf_size 10 --save_step 1`
  Build MA-COIR index for HPO (encode concepts, build graph, partition, export mappings).

We provide the SSI used for our experiments in `logs/macoir/mm-go/ssi_id_sid.json`
and `logs/macoir/mm-hpo/ssi_id_sid.json` for fast result validation.

#### Reuse embeddings (skip encoding)

- `python -m src.mining.macoir_index --ontology_file data/mm-go/meta/go-basic.json --concept_file data/mm-go/meta/biological_process_concept.json --concept_id_file data/mm-go/meta/biological_process_concept_id_list.json --embedding_pickle logs/macoir/mm-go/embeddings.pkl --graph_pickle logs/macoir/mm-go/graph.pkl --partitioned_graph_pickle logs/macoir/mm-go/graph_partitioned.pkl --checkpoint_dir logs/macoir/mm-go/checkpoints --csv_path logs/macoir/mm-go/search_id.tsv --indexing_path logs/macoir/mm-go/ssi_id_sid.json --use_semantic_similarity --L 10 --M 10 --target_leaf_size 10 --save_step 1 --skip_embed`

Build MA-COIR index for GO without recomputing embeddings.

Args:
--skip_embed (reuse embedding_pickle), other args
same as above.

- `python -m src.mining.macoir_index --ontology_file data/mm-hpo/meta/hp.json --concept_file data/mm-hpo/meta/phenotypic_abnormality_concept.json --concept_id_file data/mm-hpo/meta/phenotypic_abnormality_concept_id_list.json --embedding_pickle logs/macoir/mm-hpo/embeddings.pkl --graph_pickle logs/macoir/mm-hpo/graph.pkl --partitioned_graph_pickle logs/macoir/mm-hpo/graph_partitioned.pkl --checkpoint_dir logs/macoir/mm-hpo/checkpoints --csv_path logs/macoir/mm-hpo/search_id.tsv --indexing_path logs/macoir/mm-hpo/ssi_id_sid.json --use_semantic_similarity --L 10 --M 10 --target_leaf_size 10 --save_step 1 --skip_embed`

Build MA-COIR index for HPO without recomputing embeddings.

### Step 2: Preprocessing data

- `python -m src.datasets.macoir preprocess --data-name go --data-root data --sid-map logs/macoir/mm-go/ssi_id_sid.json`
  Build MA-COIR training JSONL files for GO from MedMentions data and search IDs.
- `python -m src.datasets.macoir preprocess --data-name hpo --data-root data --sid-map logs/macoir/mm-hpo/ssi_id_sid.json`
  Build MA-COIR training JSONL files for HPO from MedMentions data and search IDs.

### Step 3: Model training

For MA-COIR training, we construct two types of supervision.
First, each training passage is paired with its associated gold concept indices, forming the standard passage-to-index
generation instances.
Second, for every concept appearing in the training set, we additionally create concept-only instances that map the
concept name directly to its SSI.

If you prefer to not use concept-to-index instances, please remove `--augment_term_spans` from the command.

- `python -m src.trainers.macoir --train_jsonl data/mm-go/mld/train_ssi_macoir.jsonl --dev_jsonl data/mm-go/mld/dev_ssi_macoir.jsonl --sid_catalog_json logs/macoir/mm-go/ssi_id_sid.json --output_dir logs/macoir/mm-go/ckpt --transformer_name facebook/bart-large --augment_term_spans --epochs 30 --batch_size 4 --lr 1e-5 --max_src_len 1024 --max_tgt_len 300 --gen_max_len 300`

Train MA-COIR for GO.

Args:
--train_jsonl/--dev_jsonl (training/dev JSONL),
--sid_catalog_json (valid SID filter),

--output_dir (checkpoints),
--transformer_name (PLM),
--epochs/--batch_size/--lr (training hyperparams),

--max_src_len/--max_tgt_len (token limits),
--gen_max_len (decode limit).

- `python -m src.trainers.macoir --train_jsonl data/mm-hpo/mld/train_ssi_macoir.jsonl --dev_jsonl data/mm-hpo/mld/dev_ssi_macoir.jsonl --sid_catalog_json logs/macoir/mm-hpo/ssi_id_sid.json --output_dir logs/macoir/mm-hpo/ckpt --transformer_name facebook/bart-large --augment_term_spans --epochs 30 --batch_size 4 --lr 1e-5 --max_src_len 1024 --max_tgt_len 300 --gen_max_len 300`

Train MA-COIR for HPO.

### Step 4: Candidate generation

In this step, we need to conduct beam search with a beam size of 10 to get the model's prediction on the training set.

- `python -m src.evaluation.macoir_predict predict --dataset mm-go --model_dir logs/macoir/mm-go/ckpt --input_jsonl data/mm-go/mld/train_ssi_macoir.jsonl --output_jsonl logs/macoir/mm-go/model_outputs/train-beam-10-predictions.jsonl --sid_catalog_json logs/macoir/mm-go/ssi_id_sid.json`
- `python -m src.evaluation.macoir_predict predict --dataset mm-hpo --model_dir logs/macoir/mm-hpo/ckpt --input_jsonl data/mm-hpo/mld/train_ssi_macoir.jsonl --output_jsonl logs/macoir/mm-hpo/model_outputs/train-beam-10-predictions.jsonl --sid_catalog_json logs/macoir/mm-hpo/ssi_id_sid.json`

- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-go/model_outputs/train-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-go/ssi_id_sid.json --concept-meta-json data/mm-go/meta/biological_process_concept.json --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --split train --out-dir data/mm-go/mld/macoir-output`
- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-hpo/model_outputs/train-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-hpo/ssi_id_sid.json --concept-meta-json data/mm-hpo/meta/phenotypic_abnormality_concept.json --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --split train --out-dir data/mm-hpo/mld/macoir-output`

### Additional evaluation

For better assessment of model's performance, we report performance by two decoding variants.

* Beam-tuned (top-1): we select a beam width (from 1 to 10) on the development set by maximising the F1 score of the
  top-1 decoded output, and construct final predictions from the concepts mapped from that top-1 sequence at test time.
* Beam-10: we decode with a fixed beam width 10, map the top-10 decoded sequences to concepts, and take the union of the
  mapped concepts as final predictions.

Before running the commands, please make sure `logs/macoir/mm-go/model_outputs` and `logs/macoir/mm-hpo/model_outputs`
directories
exist.

- `python -m src.evaluation.macoir_predict eval --dataset mm-go --model_dir logs/macoir/mm-go/ckpt --dev_jsonl data/mm-go/mld/dev_ssi_macoir.jsonl --test_jsonl data/mm-go/mld/test_ssi_macoir.jsonl --sid_catalog_json logs/macoir/mm-go/ssi_id_sid.json --metrics_out dev-and-test-metric.json`
- `python -m src.evaluation.macoir_predict eval --dataset mm-hpo --model_dir logs/macoir/mm-hpo/ckpt --dev_jsonl data/mm-hpo/mld/dev_ssi_macoir.jsonl --test_jsonl data/mm-hpo/mld/test_ssi_macoir.jsonl --sid_catalog_json logs/macoir/mm-hpo/ssi_id_sid.json --metrics_out dev-and-test-metric.json`

Then, we conduct the post-processing and get final evaluation results.

Noticed that `--mode` can only be  `"beam10", "tuned1"`.

On MM-GO:

For Beam-10 predictions:

- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-go/model_outputs/dev-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-go/ssi_id_sid.json --concept-meta-json data/mm-go/meta/biological_process_concept.json --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --split dev --out-dir data/mm-go/mld/macoir-output`
- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-go/model_outputs/test-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-go/ssi_id_sid.json --concept-meta-json data/mm-go/meta/biological_process_concept.json --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --split test --out-dir data/mm-go/mld/macoir-output`

- `python -m src.datasets.macoir eval --dev-json data/mm-go/mld/macoir-output/dev_beam10.json --test-json data/mm-go/mld/macoir-output/test_beam10.json --out-dir data/mm-go/mld/macoir-output --tag beam10`

For Beam-tuned-top-1 predictions:

- `python -m src.datasets.macoir postprocess --mode tuned1 --pred-jsonl logs/macoir/mm-go/model_outputs/dev-beam-tuned-top-1-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-go/ssi_id_sid.json --concept-meta-json data/mm-go/meta/biological_process_concept.json --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --split dev --out-dir data/mm-go/mld/macoir-output`
- `python -m src.datasets.macoir postprocess --mode tuned1 --pred-jsonl logs/macoir/mm-go/model_outputs/test-beam-tuned-top-1-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-go/ssi_id_sid.json --concept-meta-json data/mm-go/meta/biological_process_concept.json --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --split test --out-dir data/mm-go/mld/macoir-output`

- `python -m src.datasets.macoir eval --dev-json data/mm-go/mld/macoir-output/dev_tuned1.json --test-json data/mm-go/mld/macoir-output/test_tuned1.json --out-dir data/mm-go/mld/macoir-output --tag tuned1`

On MM-HPO:

For Beam-10 predictions:

- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-hpo/model_outputs/dev-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-hpo/ssi_id_sid.json --concept-meta-json data/mm-hpo/meta/phenotypic_abnormality_concept.json --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --split dev --out-dir data/mm-hpo/mld/macoir-output`
- `python -m src.datasets.macoir postprocess --mode beam10 --pred-jsonl logs/macoir/mm-hpo/model_outputs/test-beam-10-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-hpo/ssi_id_sid.json --concept-meta-json data/mm-hpo/meta/phenotypic_abnormality_concept.json --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --split test --out-dir data/mm-hpo/mld/macoir-output`

- `python -m src.datasets.macoir eval --dev-json data/mm-hpo/mld/macoir-output/dev_beam10.json --test-json data/mm-hpo/mld/macoir-output/test_beam10.json --out-dir data/mm-hpo/mld/macoir-output --tag beam10`

For Beam-tuned-top-1 predictions:

- `python -m src.datasets.macoir postprocess --mode tuned1 --pred-jsonl logs/macoir/mm-hpo/model_outputs/dev-beam-tuned-top-1-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-hpo/ssi_id_sid.json --concept-meta-json data/mm-hpo/meta/phenotypic_abnormality_concept.json --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --split dev --out-dir data/mm-hpo/mld/macoir-output`
- `python -m src.datasets.macoir postprocess --mode tuned1 --pred-jsonl logs/macoir/mm-hpo/model_outputs/test-beam-tuned-top-1-predictions.jsonl --ssi-id-sid-json logs/macoir/mm-hpo/ssi_id_sid.json --concept-meta-json data/mm-hpo/meta/phenotypic_abnormality_concept.json --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --split test --out-dir data/mm-hpo/mld/macoir-output`

- `python -m src.datasets.macoir eval --dev-json data/mm-hpo/mld/macoir-output/dev_tuned1.json --test-json data/mm-hpo/mld/macoir-output/test_tuned1.json --out-dir data/mm-hpo/mld/macoir-output --tag tuned1`

- **Results**

On MM-GO:

| Variant            | Set  |  TP |   FP |  FN | Precision |  Recall |      F1 |
|--------------------|------|----:|-----:|----:|----------:|--------:|--------:|
| Beam-tuned (top-1) | dev  |  93 |  221 | 699 |   29.6178 | 11.7424 | 16.8174 |
|                    | test |  91 |  223 | 669 |   28.9809 | 11.9737 | 16.9460 |
| Beam-10            | dev  | 126 | 1273 | 666 |    9.0064 | 15.9091 | 11.5016 |
|                    | test | 119 | 1272 | 641 |    8.5550 | 15.6579 | 11.0646 |

On MM-HPO:

| Variant            | Set  |  TP |   FP |  FN | Precision |  Recall |      F1 |
|--------------------|------|----:|-----:|----:|----------:|--------:|--------:|
| Beam-tuned (top-1) | dev  | 193 |  218 | 713 |   46.9586 | 21.3024 | 29.3090 |
|                    | test | 178 |  232 | 719 |   43.4146 | 19.8439 | 27.2379 |
| Beam-10            | dev  | 232 | 1702 | 674 |   11.9959 | 25.6071 | 16.3380 |
|                    | test | 237 | 1625 | 660 |   12.7282 | 26.4214 | 17.1801 |

## XR-Transformer

As we used the original implementation of XR-Transformer provided by https://github.com/amzn/pecos, we only provide the
script for preprocessing and postprocessing.

### Step 1: Preprocessing data

- `python -m src.datasets.xrt preprocess --data-root data --data-name go --concepts biological_process_concept_id_list.json --id_map_out xrt_label_to_concept.json`
- `python -m src.datasets.xrt preprocess --data-root data --data-name hpo --concepts phenotypic_abnormality_concept_id_list.json --id_map_out xrt_label_to_concept.json`

### Step 2: Model training

As you download the library **pecos** from  https://github.com/amzn/pecos, please copy the preprocessed
data (`data/mm-go/mld/xrt-input` and `data/mm-hpo/mld/xrt-input`) to **pecos**, conduct the model training.

### Step 3: Candidate generation

In this step, we use the trained model to provided top-100 labels (concepts) on the training set first.

After you conduct prediction by **pecos**, you will find the prediction files are hardly to read. There is a need to
postprocess them.
Please copy these files into `logs/xrt/mm-go` and `logs/xrt/mm-hpo`, change their name to `train/dev/test`,
then run the postprocessing commands.

- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name go --id_map_path xrt_label_to_concept.json --id_name_path biological_process_concept.json --model_outputs train`
- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name hpo --id_map_path xrt_label_to_concept.json --id_name_path phenotypic_abnormality_concept.json --model_outputs train`

We provide the top-100 predictions of XR-Transformer in `logs/xrt/mm-go/top-100-predictions`
and `logs/xrt/mm-hpo/top-100-predictions` to help fast result verification.

To ensure fair comparison across recognizers, we match the average number of false positives per passage across models:
we use MA-COIR (Beam-10) as a reference and adjust the thresholds of XR-Transformer and Bi-Encoder so that all three
recognizers produce the same average number of false positives per passage on the training set.
False positives produced under this matched setting are collected as candidates.

### Additional evaluation

For better assessment of model's performance, we report performance by two decoding variants.

* Thresholded: Include all concepts whose scores exceed a decision threshold tuned on the development set to maximise
  Micro-F1.
* Top-100: Take the top-$L$ ranked concepts without thresholding.

- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name go --id_map_path xrt_label_to_concept.json --id_name_path biological_process_concept.json --model_outputs dev`
- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name go --id_map_path xrt_label_to_concept.json --id_name_path biological_process_concept.json --model_outputs test`
  Postprocess model's predictions on development and test set of MM-GO.

- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name hpo --id_map_path xrt_label_to_concept.json --id_name_path phenotypic_abnormality_concept.json --model_outputs dev`
- `python -m src.datasets.xrt postprocess --data-root logs/xrt --data-name hpo --id_map_path xrt_label_to_concept.json --id_name_path phenotypic_abnormality_concept.json --model_outputs test`
  Postprocess model's predictions on development and test set of MM-HPO.

We provide the top-100 predictions of XR-Transformer in `logs/xrt/mm-go/top-100-predictions`
and `logs/xrt/mm-hpo/top-100-predictions` to help fast result verification.

Run following commands to get evaluation results.

- `python -m src.evaluation.ranker_top100_eval --mode topk --topk 100 --dev-jsonl logs/xrt/mm-hpo/top-100-predictions/dev.jsonl --test-jsonl logs/xrt/mm-hpo/top-100-predictions/test.jsonl --train-jsonl logs/xrt/mm-hpo/top-100-predictions/train.jsonl --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --out-dir data/mm-hpo/mld/xrt-output --with-score`

Keeps the top-100 candidates as-is for each query (dev/test/train), computes set-based micro P/R/F1, and writes
prediction dumps + metrics.

Args:
--mode topk: use top-k selection

--topk 100: keep the top 100 candidates

--dev-jsonl, --test-jsonl, --train-jsonl: model outputs in the shared top-K JSONL format

--gold-json: gold file (all_wo_mention.json, used to fetch gold concept IDs per document)

--split-list-json: split definition (split_list.json, mapping doc IDs to train/dev/test)

--out-dir: output directory (writes dev_top100.json, test_top100.json, train_top100.json, predictions_all_top100.json,
metrics_top100.json)

`python -m src.evaluation.ranker_top100_eval --mode thresholded --topk 100 --dev-jsonl logs/xrt/mm-hpo/top-100-predictions/dev.jsonl --test-jsonl logs/xrt/mm-hpo/top-100-predictions/test.jsonl --train-jsonl logs/xrt/mm-hpo/top-100-predictions/train.jsonl --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --out-dir data/mm-hpo/mld/xrt-output --with-score`

Scans thresholds on dev (within the top-100 candidate pool) to maximize set-micro F1, then filters candidates by the
best threshold for test/train, and writes prediction dumps + metrics.

Args:

--mode thresholded: tune a score threshold on dev and apply it to other splits

--topk 100: thresholding is applied within the top-100 candidate pool

(all other arguments are the same as in Top-100 mode)

Add this flag to either command: `--with-score`, to adds a score field for each predicted concept in the output JSON.
In default, we output scores for further distribution matching step.

You can replace `mm-hpo` with `mm-go` in the commands to get the evaluated results.

**Results**

On MM-HPO:

| Variant                  | Set  |  TP |    FP |  FN | Precision |  Recall |      F1 |
|--------------------------|------|----:|------:|----:|----------:|--------:|--------:|
| Top-100                  | dev  | 431 | 43769 | 475 |    0.9751 | 47.5717 |  1.9111 |
|                          | test | 403 | 41197 | 492 |    0.9688 | 45.0279 |  1.8967 |
| Tuned threshold=0.048433 | dev  | 307 |   320 | 599 |   48.9633 | 33.8852 | 40.0522 |
|                          | test | 276 |   326 | 619 |   45.8472 | 30.8380 | 36.8737 |

On MM-GO:

| Variant                  | Set  |  TP |    FP |  FN | Precision |  Recall |      F1 |
|--------------------------|------|----:|------:|----:|----------:|--------:|--------:|
| Top-100                  | dev  | 334 | 33266 | 458 |    0.9940 | 42.1717 |  1.9423 |
|                          | test | 333 | 34667 | 427 |    0.9514 | 43.8158 |  1.8624 |
| Tuned threshold=0.045884 | dev  | 204 |   354 | 588 |   36.5591 | 25.7576 | 30.2222 |
|                          | test | 202 |   339 | 558 |   37.3383 | 26.5789 | 31.0530 |

## Bi-Encoder

### Step 1: Preprocessing data

For the bi-encoder training, we use both hard negatives derived by BM25 and in-batch negatives.

`python -m src.datasets.bi_encoder preprocess --data-root data --data-name go --hard-negative-path logs/bm25/mm-go_mld_all_concept.jsonl --k-hard-neg 20`
`python -m src.datasets.bi_encoder preprocess --data-root data --data-name hpo --hard-negative-path logs/bm25/mm-hpo_mld_all_concept.jsonl --k-hard-neg 20`

Corresponding data is placed in `data/mm-go/mld/bi-encoder-input/` and `data/mm-hpo/mld/bi-encoder-input/`.

### Step 2: Model training

`python -m src.trainers.bi_encoder --train_jsonl data/mm-go/mld/bi-encoder-input/train_biencoder.jsonl --eval_jsonl data/mm-go/mld/bi-encoder-input/dev_biencoder.jsonl --model_name cambridgeltl/SapBERT-from-PubMedBERT-fulltext --output_dir logs/bi_encoder/mm-go/ckpt --epochs 5 --batch_size 16 --lr 2e-5 --hard_neg_k 2`
`python -m src.trainers.bi_encoder --train_jsonl data/mm-hpo/mld/bi-encoder-input/train_biencoder.jsonl --eval_jsonl data/mm-hpo/mld/bi-encoder-input/dev_biencoder.jsonl --model_name cambridgeltl/SapBERT-from-PubMedBERT-fulltext --output_dir logs/bi_encoder/mm-hpo/ckpt --epochs 5 --batch_size 16 --lr 2e-5 --hard_neg_k 2`

### Step 3: Candidate generation

In this step, we use the trained model to provided top-100 labels (concepts) on the training set first.

To ensure fair comparison across recognizers, we match the average number of false positives per passage across models:
we use MA-COIR (Beam-10) as a reference and adjust the thresholds of XR-Transformer and Bi-Encoder so that all three
recognizers produce the same average number of false positives per passage on the training set.
False positives produced under this matched setting are collected as candidates.

### Additional evaluation

For better assessment of model's performance, we report performance by two decoding variants.

* Thresholded: Include all concepts whose scores exceed a decision threshold tuned on the development set to maximise
  Micro-F1.
* Top-100: Take the top-$L$ ranked concepts without thresholding.

- `python -m src.evaluation.ranker_top100_eval --mode topk --topk 100 --dev-jsonl logs/bi_encoder/mm-go/top-100-predictions/dev.jsonl --test-jsonl logs/bi_encoder/mm-go/top-100-predictions/test.jsonl --train-jsonl logs/bi_encoder/mm-go/top-100-predictions/train.jsonl --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --out-dir data/mm-go/mld/bi-encoder-output --with-score`
- `python -m src.evaluation.ranker_top100_eval --mode thresholded --topk 100 --dev-jsonl logs/bi_encoder/mm-go/top-100-predictions/dev.jsonl --test-jsonl logs/bi_encoder/mm-go/top-100-predictions/test.jsonl --train-jsonl logs/bi_encoder/mm-go/top-100-predictions/train.jsonl --gold-json data/mm-go/ori/all_wo_mention.json --split-list-json data/mm-go/ori/split_list.json --out-dir data/mm-go/mld/bi-encoder-output --with-score`


- `python -m src.evaluation.ranker_top100_eval --mode topk --topk 100 --dev-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/dev.jsonl --test-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/test.jsonl --train-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/train.jsonl --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --out-dir data/mm-hpo/mld/bi-encoder-output --with-score`
- `python -m src.evaluation.ranker_top100_eval --mode thresholded --topk 100 --dev-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/dev.jsonl --test-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/test.jsonl --train-jsonl logs/bi_encoder/mm-hpo/top-100-predictions/train.jsonl --gold-json data/mm-hpo/ori/all_wo_mention.json --split-list-json data/mm-hpo/ori/split_list.json --out-dir data/mm-hpo/mld/bi-encoder-output --with-score`

**Results**

On MM-HPO:

Our default setting `epochs=5`:

| Variant                  | Set  |  TP |    FP |  FN | Precision |  Recall |      F1 |
|--------------------------|------|----:|------:|----:|----------:|--------:|--------:|
| Top-100                  | dev  | 835 | 43365 |  71 |    1.8891 | 92.1634 |  3.7024 |
|                          | test | 777 | 40923 | 120 |    1.8633 | 86.6221 |  3.6481 |
| Tuned threshold=0.594208 | dev  | 362 |   588 | 544 |   38.1053 | 39.9558 | 39.0086 |
|                          | test | 329 |   618 | 568 |   34.7413 | 36.6778 | 35.6833 |

On MM-GO:

Our default setting `epochs=5`:

| Variant                  | Set  |  TP |    FP |  FN | Precision |  Recall |      F1 |
|--------------------------|------|----:|------:|----:|----------:|--------:|--------:|
| Top-100                  | dev  | 567 | 33033 | 225 |    1.6875 | 71.5909 |  3.2973 |
|                          | test | 567 | 34433 | 193 |    1.6200 | 74.6053 |  3.1711 |
| Tuned threshold=0.549273 | dev  | 169 |   701 | 623 |   19.4253 | 21.3384 | 20.3369 |
|                          | test | 147 |   610 | 613 |   19.4188 | 19.3421 | 19.3804 |

If `epochs=20`:

| Variant                  | Set  |  TP |    FP |  FN | Precision |  Recall |      F1 |
|--------------------------|------|----:|------:|----:|----------:|--------:|--------:|
| Top-100                  | dev  | 598 | 33002 | 194 |    1.7798 | 75.5051 |  3.4776 |
|                          | test | 594 | 34406 | 166 |    1.6971 | 78.1579 |  3.3221 |
| Tuned threshold=0.571765 | dev  | 182 |   457 | 610 |   28.4820 | 22.9798 | 25.4368 |
|                          | test | 148 |   418 | 612 |   26.1484 | 19.4737 | 22.3228 |

## Get Error-derived Negatives

### Get FP from three models

On MM-GO:

- `python -m src.datasets.cross_encoder --dataset go extract-fp --pred-dir data/mm-go/mld/macoir-output --variant beam10`
- `python -m src.datasets.cross_encoder --dataset go extract-fp --pred-dir data/mm-go/mld/bi-encoder-output --variant thresholded`
- `python -m src.datasets.cross_encoder --dataset go extract-fp --pred-dir data/mm-go/mld/bi-encoder-output --variant top100`
- `python -m src.datasets.cross_encoder --dataset go extract-fp --pred-dir data/mm-go/mld/xrt-output --variant thresholded`
- `python -m src.datasets.cross_encoder --dataset go extract-fp --pred-dir data/mm-go/mld/xrt-output --variant top100`

On MM-HPO:

- `python -m src.datasets.cross_encoder --dataset hpo extract-fp --pred-dir data/mm-hpo/mld/macoir-output --variant beam10`
- `python -m src.datasets.cross_encoder --dataset hpo extract-fp --pred-dir data/mm-hpo/mld/bi-encoder-output --variant thresholded`
- `python -m src.datasets.cross_encoder --dataset hpo extract-fp --pred-dir data/mm-hpo/mld/bi-encoder-output --variant top100`
- `python -m src.datasets.cross_encoder --dataset hpo extract-fp --pred-dir data/mm-hpo/mld/xrt-output --variant thresholded`
- `python -m src.datasets.cross_encoder --dataset hpo extract-fp --pred-dir data/mm-hpo/mld/xrt-output --variant top100`

The FP files are placed in `data/mm-go/mld/fp` and `data/mm-hpo/mld/fp`.

If there is missing split, like no prediction on dev set, you can add `--allow-missing-splits` to the commands.

### Get distribution-matched FPs

To ensure fair comparison across recognizers, we match the **average number of false positives per passage** across
models:
we use MA-COIR (Beam-10) as a reference and adjust the thresholds of XR-Transformer and Bi-Encoder so that all three
recognizers produce the same average number of false positives per passage on the training set.
False positives produced under this matched setting are collected as ENs.

On MM-GO:

- `python -m src.datasets.cross_encoder --dataset go map-to-macoir-by-fpcount --macoir-fp-json data/mm-go/mld/fp/fp_macoir-output_beam10.json --scored-pred-json data/mm-go/mld/bi-encoder-output/train_top100.json --out-fp-json data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json`
- `python -m src.datasets.cross_encoder --dataset go map-to-macoir-by-fpcount --macoir-fp-json data/mm-go/mld/fp/fp_macoir-output_beam10.json --scored-pred-json data/mm-go/mld/xrt-output/train_top100.json --out-fp-json data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json`

```
[map-to-macoir] macoir_docs=1718 scored_docs=1032 intersection=1032 dropped_macoir=686 dropped_scored=0
[map-to-macoir] target_fp_total=3262 mapped_fp_total=3261 gap=1 tol=13 best_thr=0.531008 candidates=4096 docs=1032
[map-to-macoir] wrote: data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json
[map-to-macoir] macoir_docs=1718 scored_docs=1032 intersection=1032 dropped_macoir=686 dropped_scored=0
[map-to-macoir] target_fp_total=3262 mapped_fp_total=3255 gap=7 tol=13 best_thr=0.032073 candidates=4096 docs=1032
[map-to-macoir] wrote: data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json

```

On MM-HPO:

- `python -m src.datasets.cross_encoder --dataset go map-to-macoir-by-fpcount --macoir-fp-json data/mm-hpo/mld/fp/fp_macoir-output_beam10.json --scored-pred-json data/mm-hpo/mld/bi-encoder-output/train_top100.json --out-fp-json data/mm-hpo/mld/fp/fp_be_map_to_macoir_beam10.json`
- `python -m src.datasets.cross_encoder --dataset go map-to-macoir-by-fpcount --macoir-fp-json data/mm-hpo/mld/fp/fp_macoir-output_beam10.json --scored-pred-json data/mm-hpo/mld/xrt-output/train_top100.json --out-fp-json data/mm-hpo/mld/fp/fp_xrt_map_to_macoir_beam10.json`

```
[map-to-macoir] macoir_docs=2171 scored_docs=1312 intersection=1312 dropped_macoir=859 dropped_scored=0
[map-to-macoir] target_fp_total=4145 mapped_fp_total=4148 gap=3 tol=16 best_thr=0.552371 candidates=4096 docs=1312
[map-to-macoir] wrote: data/mm-hpo/mld/fp/fp_be_map_to_macoir_beam10.json
[map-to-macoir] macoir_docs=2171 scored_docs=1312 intersection=1312 dropped_macoir=859 dropped_scored=0
[map-to-macoir] target_fp_total=4145 mapped_fp_total=4151 gap=6 tol=16 best_thr=0.034581 candidates=4096 docs=1312
[map-to-macoir] wrote: data/mm-hpo/mld/fp/fp_xrt_map_to_macoir_beam10.json
```

### Construct EN-only data

Single recognizer:

On MM-GO:

- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_macoir-output_beam10.json --out data/mm-go/mld/cross-encoder-input/train_macoir_fp.jsonl`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json --out data/mm-go/mld/cross-encoder-input/train_be_fp.jsonl`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json --out data/mm-go/mld/cross-encoder-input/train_xrt_fp.jsonl`

```
[fp-only] records=2526 avg_hard=2.18 -> data/mm-go/mld/cross-encoder-input/train_macoir_fp.jsonl
[fp-only] records=2526 avg_hard=3.57 -> data/mm-go/mld/cross-encoder-input/train_be_fp.jsonl
[fp-only] records=2526 avg_hard=4.65 -> data/mm-go/mld/cross-encoder-input/train_xrt_fp.jsonl
```

On MM-HPO:

- `python -m src.datasets.cross_encoder --dataset hpo build-fp-only --fp-json data/mm-hpo/mld/fp/fp_macoir-output_beam10.json --out data/mm-hpo/mld/cross-encoder-input/train_macoir_fp.jsonl`
- `python -m src.datasets.cross_encoder --dataset hpo build-fp-only --fp-json data/mm-hpo/mld/fp/fp_be_map_to_macoir_beam10.json --out data/mm-hpo/mld/cross-encoder-input/train_be_fp.jsonl`
- `python -m src.datasets.cross_encoder --dataset hpo build-fp-only --fp-json data/mm-hpo/mld/fp/fp_xrt_map_to_macoir_beam10.json --out data/mm-hpo/mld/cross-encoder-input/train_xrt_fp.jsonl`

```
[fp-only] records=2720 avg_hard=2.38 -> data/mm-hpo/mld/cross-encoder-input/train_macoir_fp.jsonl
[fp-only] records=2720 avg_hard=3.04 -> data/mm-hpo/mld/cross-encoder-input/train_be_fp.jsonl
[fp-only] records=2720 avg_hard=4.34 -> data/mm-hpo/mld/cross-encoder-input/train_xrt_fp.jsonl
```

Multiple Recognizer:

Firstly, we need to merge multi-resource FPs into one FP file. 

Then, we build data based on the merged FP file.

On MM-GO:

- `python -m src.datasets.cross_encoder --dataset go merge-fp --inputs data/mm-go/mld/fp/fp_macoir-output_beam10.json data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json --out data/mm-go/mld/fp/fp_macoir_be.json`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_macoir_be.json --out data/mm-go/mld/cross-encoder-input/train_macoir_be_fp.jsonl`

- `python -m src.datasets.cross_encoder --dataset go merge-fp --inputs data/mm-go/mld/fp/fp_macoir-output_beam10.json data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json --out data/mm-go/mld/fp/fp_macoir_xrt.json`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_macoir_xrt.json --out data/mm-go/mld/cross-encoder-input/train_macoir_xrt_fp.jsonl`

- `python -m src.datasets.cross_encoder --dataset go merge-fp --inputs data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json --out data/mm-go/mld/fp/fp_xrt_be.json`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_xrt_be.json --out data/mm-go/mld/cross-encoder-input/train_xrt_be_fp.jsonl`

- `python -m src.datasets.cross_encoder --dataset go merge-fp --inputs data/mm-go/mld/fp/fp_macoir-output_beam10.json data/mm-go/mld/fp/fp_be_map_to_macoir_beam10.json data/mm-go/mld/fp/fp_xrt_map_to_macoir_beam10.json --out data/mm-go/mld/fp/fp_macoir_xrt_be.json`
- `python -m src.datasets.cross_encoder --dataset go build-fp-only --fp-json data/mm-go/mld/fp/fp_macoir_xrt_be.json --out data/mm-go/mld/cross-encoder-input/train_macoir_xrt_be_fp.jsonl`

```
[fp-only] records=2526 avg_hard=5.71 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_fp.jsonl
[fp-only] records=2526 avg_hard=6.81 -> data/mm-go/mld/cross-encoder-input/train_macoir_xrt_fp.jsonl
[fp-only] records=2526 avg_hard=8.14 -> data/mm-go/mld/cross-encoder-input/train_xrt_be_fp.jsonl
[fp-only] records=2526 avg_hard=10.27 -> data/mm-go/mld/cross-encoder-input/train_macoir_xrt_be_fp.jsonl
```

On MM-HPO:

```
[fp-only] records=2720 avg_hard=5.33 -> data/mm-hpo/mld/cross-encoder-input/train_macoir_be_fp.jsonl
[fp-only] records=2720 avg_hard=6.69 -> data/mm-hpo/mld/cross-encoder-input/train_macoir_xrt_fp.jsonl
[fp-only] records=2720 avg_hard=7.30 -> data/mm-hpo/mld/cross-encoder-input/train_xrt_be_fp.jsonl
[fp-only] records=2720 avg_hard=9.57 -> data/mm-hpo/mld/cross-encoder-input/train_macoir_xrt_be_fp.jsonl
```

### Construct GN+EN data

Take GNs + ENs (MA-COIR & Bi-Encoder) on MM-GO as an example:

- `python -m src.datasets.cross_encoder --dataset go build-mix --fp-json data/mm-go/mld/fp/fp_macoir_be.json --bm25-jsonl logs/bm25/mm-go_mld_all_concept.jsonl --pool-sizes 5 10 15 20 25 30 35 40 45 50 --out-dir data/mm-go/mld/cross-encoder-input --tag macoir_be`

```
[mix] pool=5 records=2526 avg_hard=10.40 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_5.jsonl
[mix] pool=10 records=2526 avg_hard=14.99 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_10.jsonl
[mix] pool=15 records=2526 avg_hard=19.56 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_15.jsonl
[mix] pool=20 records=2526 avg_hard=24.07 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_20.jsonl
[mix] pool=25 records=2526 avg_hard=28.51 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_25.jsonl
[mix] pool=30 records=2526 avg_hard=32.87 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_30.jsonl
[mix] pool=35 records=2526 avg_hard=37.20 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_35.jsonl
[mix] pool=40 records=2526 avg_hard=41.53 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_40.jsonl
[mix] pool=45 records=2526 avg_hard=45.81 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_45.jsonl
[mix] pool=50 records=2526 avg_hard=50.08 -> data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_50.jsonl

```

You can change `--dataset`, `--fp-json` and `--tag` to build corresponding data.

### Count average number of hard negatives per query

- `python -m src.datasets.cross_encoder --dataset go stats-hard --train-jsonl data/mm-go/mld/cross-encoder-input/train_macoir_be_gold_bm25_20.jsonl`

