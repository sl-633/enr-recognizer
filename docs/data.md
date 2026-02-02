## Ontology

Official file can be downloaded from:

- Gene Ontology (GO)  https://purl.obolibrary.org/obo/go/go-basic.json
- Human Phenotype Ontology (HPO) v2025-10-22 https://hpo.jax.org/data/ontology

Put the ontology file of HPO into `data/mm-hpo/meta/` as `hp.json`.
Put the ontology file of GO into `data/mm-go/meta` as `go-basic.json`.

### Data prep commands

- `python -m src.datasets.ontology --data-root data`: Generate GO/HPO descendant ID lists, concept name/synonym JSON, and ontology JSONL files for BM25 mining under `mm-go/meta` and `mm-hpo/meta`.

### Statistics:

| Item            | GO     | HPO    |
|-----------------|--------|--------|
| Concept         | 48,030 | 19,763 |
| Target Concept  | 26,036 | 18,585 |
| Name & Synonyms | 84,022 | 42,230 |

## MedMention-ST21pv

Download and put these 5 files into `data/mm-st21pv`.
* corpus_pubtator.txt
* corpus_pubtator_pmids_dev.txt
* corpus_pubtator_pmids_test.txt
* corpus_pubtator_pmids_trng.txt
* MRCONSO.RRF

### Data prep commands

- `python -m src.datasets.medmention --data-root data --mrconso-path data/mm-st21pv/MRCONSO.RRF`: Parse MedMentions PubTator data, map UMLS→GO/HPO, collapse mentions, build splits/unseen stats, and export BM25 JSONL files under `mm-go` and `mm-hpo`.

### Statistics

MedMentions-st21pv (MM-st21pv)

| Set                    | Passage | Mention | Concept | Unique Concept |
|------------------------|---------|---------|---------|----------------|
| MM-st21pv              | 4,392   | 203,282 | 98,596  | 25,419         |
| MM-GO (MM-st21pv-GO)   | 1,718   | 7,635   | 4,708   | 1,002          |
| MM-HPO (MM-st21pv-HPO) | 2,171   | 11,705  | 4,523   | 1,238          |

Passage

| Set                     | Train | Development | Test |
|-------------------------|-------|-------------|------|
| MM-GO (MM-st21pv-GO)    | 1,032 | 336         | 350  |
| MM-HPO (MM-st21pv-HPO)  | 1,312 | 442         | 417  |

Unique Concept

| Set                    | Train | Development | Test | Unseen in Dev | Unseen in Test (%) |
|------------------------|-------|-------------|------|---------------|--------------------|
| MM-GO (MM-st21pv-GO)   | 746   | 335         | 336  | 140 (41.8%)   | 126 (37.5%)        |
| MM-HPO (MM-st21pv-HPO) | 949   | 455         | 440  | 149 (32.7%)   | 152 (34.5%)        |

