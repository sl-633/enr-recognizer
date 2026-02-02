"""Preprocessing utilities for the MedMentions (st21pv) dataset."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

BASE_IRI = "http://purl.obolibrary.org/obo/"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MedMentionPaths:
    """Collection of common MedMentions paths for preprocessing."""

    data_root: Path

    @property
    def st21pv_dir(self) -> Path:
        return self.data_root / "mm-st21pv"

    @property
    def go_dir(self) -> Path:
        return self.data_root / "mm-go"

    @property
    def hpo_dir(self) -> Path:
        return self.data_root / "mm-hpo"

    @property
    def st21pv_corpus(self) -> Path:
        return self.st21pv_dir / "corpus_pubtator.txt"

    def st21pv_split(self, split: str) -> Path:
        return self.st21pv_dir / f"corpus_pubtator_pmids_{split}.txt"


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_umls_mappings(mrconso_path: Path, output_dir: Path) -> dict[str, dict[str, str]]:
    """Parse MRCONSO.RRF and export UMLS-to-ontology mappings."""
    umls2hpo: dict[str, str] = {}
    umls2go: dict[str, str] = {}
    umls2sct: dict[str, str] = {}

    with mrconso_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            elements = line.strip().split("|")
            if len(elements) < 12:
                continue
            cui = elements[0]
            code = elements[10]
            sab = elements[11]

            if sab == "HPO" and code.startswith("HP:"):
                umls2hpo[cui] = code
            elif sab == "GO" and code.startswith("GO:"):
                umls2go[cui] = code
            elif sab == "SNOMEDCT_US":
                umls2sct[cui] = elements[9]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "umls2go.json", umls2go)
    write_json(output_dir / "umls2hpo.json", umls2hpo)
    write_json(output_dir / "umls2sct.json", umls2sct)

    LOGGER.info("GO terms: %s", len(umls2go))
    LOGGER.info("HPO terms: %s", len(umls2hpo))
    LOGGER.info("SNOMEDCT terms: %s", len(umls2sct))
    return {"go": umls2go, "hpo": umls2hpo, "sct": umls2sct}


def parse_pubtator_corpus(corpus_path: Path, output_path: Path) -> dict[str, dict[str, object]]:
    """Parse PubTator-formatted MedMentions corpus."""
    all_data: dict[str, dict[str, object]] = {}
    all_appeared_id: set[str] = set()
    count_concept = 0

    current_id = ""
    current_title = ""
    current_passage = ""
    current_mentions: list[dict[str, object]] = []
    current_unique_ids: set[str] = set()

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if "|t|" in line:
                elements = line.split("|")
                current_id = elements[0]
                current_title = elements[-1]
            elif "|a|" in line:
                elements = line.split("|")
                if current_id != elements[0]:
                    raise ValueError("PubTator parse mismatch between title and abstract IDs")
                current_passage = elements[-1]
            elif not line:
                all_data[current_id] = {
                    "title": current_title,
                    "passage": current_passage,
                    "concepts": current_mentions,
                }
                count_concept += len(current_unique_ids)
                current_id = ""
                current_title = ""
                current_passage = ""
                current_mentions = []
                current_unique_ids.clear()
            else:
                elements = line.split("\t")
                if current_id != elements[0]:
                    raise ValueError("PubTator parse mismatch between mention and document IDs")
                umls_id = elements[-1].replace("UMLS:", "")
                current_mentions.append(
                    {
                        "umls_id": umls_id,
                        "offset": [int(elements[1]), int(elements[2])],
                        "mention": elements[3],
                        "umls_type": elements[4],
                    }
                )
                all_appeared_id.add(elements[-1])
                current_unique_ids.add(elements[-1])

    write_json(output_path, all_data)
    LOGGER.info("Number of passages in MM-st21pv: %s", len(all_data))
    LOGGER.info("Number of concepts in MM-st21pv: %s", count_concept)
    LOGGER.info("Number of unique concepts in MM-st21pv: %s", len(all_appeared_id))
    return all_data


def map_umls_to_ontology_with_mentions(
    source_path: Path,
    mapping_path: Path,
    id_list_path: Path,
    output_path: Path,
    ontology_key: str,
) -> dict[str, dict[str, object]]:
    """Map UMLS mentions to ontology IDs and filter by target ontology list."""
    umls_map = load_json(mapping_path)
    all_data = load_json(source_path)
    id_list = set(load_json(id_list_path))

    new_data: dict[str, dict[str, object]] = {}
    count_all_mentions = 0
    count_onto_mentions = 0
    count_no_ent_docs = 0

    for doc_id, doc in all_data.items():
        valid_mentions = []
        for mention in doc["concepts"]:
            umls_id = mention["umls_id"]
            if umls_id not in umls_map:
                continue
            onto_id = umls_map[umls_id].replace(":", "_")
            if onto_id not in id_list:
                continue
            enriched = dict(mention)
            enriched[ontology_key] = onto_id
            valid_mentions.append(enriched)

        count_onto_mentions += len(valid_mentions)
        count_all_mentions += len(doc["concepts"])
        if not valid_mentions:
            count_no_ent_docs += 1
            continue
        new_doc = dict(doc)
        new_doc["concepts"] = valid_mentions
        new_data[doc_id] = new_doc

    write_json(output_path, new_data)
    LOGGER.info("Mapping annotation with: %s", mapping_path)
    LOGGER.info("Number of passages: %s", len(new_data))
    LOGGER.info("Number of mapped mentions: %s", count_onto_mentions)
    LOGGER.info("Number of invalid passages: %s", count_no_ent_docs)
    LOGGER.info(
        "Number of invalid mentions: %s (ALL: %s)",
        count_all_mentions - count_onto_mentions,
        count_all_mentions,
    )
    return new_data


def collapse_mentions_to_concepts(
    source_path: Path,
    ontology_path: Path,
    id_list_path: Path,
    output_path: Path,
    ontology_key: str,
) -> dict[str, dict[str, object]]:
    """Collapse mention-level annotations into unique concept lists per document."""
    all_data = load_json(source_path)
    onto_info = load_json(ontology_path)
    id_list = set(load_json(id_list_path))

    new_data: dict[str, dict[str, object]] = {}
    count_all_mentions = 0
    count_unique_concepts = 0
    count_no_ent_docs = 0
    all_appeared_id: set[str] = set()

    for doc_id, doc in all_data.items():
        valid_concepts = []
        appeared = set()
        for mention in doc["concepts"]:
            onto_id = mention[ontology_key]
            if onto_id not in id_list or onto_id in appeared:
                continue
            valid_concepts.append({"id": onto_id, "name": onto_info[onto_id]["name"]})
            appeared.add(onto_id)
            all_appeared_id.add(onto_id)
        count_unique_concepts += len(valid_concepts)
        count_all_mentions += len(doc["concepts"])
        if not valid_concepts:
            count_no_ent_docs += 1
            continue
        new_doc = dict(doc)
        new_doc["concepts"] = valid_concepts
        new_data[doc_id] = new_doc

    write_json(output_path, new_data)
    LOGGER.info("Processing: %s", ontology_key)
    LOGGER.info("Number of passages: %s", len(new_data))
    LOGGER.info("Number of concepts: %s", count_unique_concepts)
    LOGGER.info("Number of unique concepts: %s", len(all_appeared_id))
    return new_data


def build_split_lists(
    data_name: str,
    data_root: Path,
    output_path: Path,
) -> dict[str, list[str]]:
    """Generate train/dev/test document lists for MM-{data_name}."""
    all_data = load_json(data_root / f"mm-{data_name}" / "ori" / "all_wo_mention.json")

    split_map: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    LOGGER.info("Number of passages in MM-st21pv-%s:", data_name.upper())

    for split in ["trng", "dev", "test"]:
        split_file = data_root / "mm-st21pv" / f"corpus_pubtator_pmids_{split}.txt"
        doc_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
        selected = [doc_id for doc_id in all_data if doc_id in doc_ids and all_data[doc_id]["concepts"]]
        LOGGER.info("%s\t%s", split, len(selected))
        if split == "trng":
            split_map["train"] = selected
        else:
            split_map[split] = selected

    write_json(output_path, split_map)
    return split_map


def compute_unseen_statistics(data_name: str, data_root: Path) -> dict[str, list[str]]:
    """Compute unseen concept statistics for dev/test splits."""
    all_data = load_json(data_root / f"mm-{data_name}" / "ori" / "all_wo_mention.json")
    split_list = load_json(data_root / f"mm-{data_name}" / "ori" / "split_list.json")

    LOGGER.info("Number of concepts in MM-st21pv-%s:", data_name.upper())

    concepts: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for split in ["train", "dev", "test"]:
        for doc_id in split_list[split]:
            for concept in all_data[doc_id]["concepts"]:
                concepts[split].add(concept["id"])
        LOGGER.info("%s\t%s", split, len(concepts[split]))

    unseen_in_dev = concepts["dev"] - concepts["train"]
    unseen_in_test = concepts["test"] - concepts["train"]
    LOGGER.info("Processing: %s", data_name)
    LOGGER.info("Number of unseen concepts in dev: %s", len(unseen_in_dev))
    LOGGER.info("Number of unseen concepts in test: %s", len(unseen_in_test))

    write_json(
        data_root / f"mm-{data_name}" / "ori" / "unseen_list.json",
        {"dev": sorted(unseen_in_dev), "test": sorted(unseen_in_test)},
    )
    write_json(
        data_root / f"mm-{data_name}" / "ori" / "in_data_concept_list.json",
        {split: sorted(values) for split, values in concepts.items()},
    )
    return {"dev": sorted(unseen_in_dev), "test": sorted(unseen_in_test)}


def export_bm25_jsonl(data_name: str, data_root: Path) -> None:
    """Export passage and concept JSONL files for BM25 retrieval."""
    all_data = load_json(data_root / f"mm-{data_name}" / "ori" / "all_wo_mention.json")

    unique_concepts: set[tuple[str, str]] = set()
    passage_output = data_root / f"mm-{data_name}" / "mld" / "all_passage_for_bm25.jsonl"
    passage_output.parent.mkdir(parents=True, exist_ok=True)

    with passage_output.open("w", encoding="utf-8") as out:
        for doc_id, doc in all_data.items():
            record = {
                "doc_id": doc_id,
                "text": doc["passage"],
                "positive_concept_ids": [c["id"] for c in doc["concepts"]],
            }
            for concept in doc["concepts"]:
                unique_concepts.add((concept["id"], concept["name"]))
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Finished transformation, output: %s", passage_output)

    concept_output = data_root / f"mm-{data_name}" / "mld" / "all_concept_for_bm25.jsonl"
    with concept_output.open("w", encoding="utf-8") as out:
        for concept_id, concept_name in sorted(unique_concepts):
            record = {
                "doc_id": concept_id,
                "text": concept_name,
                "positive_concept_ids": [concept_id],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Finished transformation, output: %s", concept_output)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess MedMentions (st21pv) data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../data"),
        help="Root directory containing mm-st21pv, mm-go, and mm-hpo datasets.",
    )
    parser.add_argument(
        "--mrconso-path",
        type=Path,
        default=None,
        help="Path to MRCONSO.RRF for UMLS mappings (optional).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    paths = MedMentionPaths(data_root=args.data_root)

    if args.mrconso_path:
        parse_umls_mappings(args.mrconso_path, paths.st21pv_dir)

    parse_pubtator_corpus(
        corpus_path=paths.st21pv_corpus,
        output_path=paths.st21pv_dir / "all_to_umls_w_mention.json",
    )

    map_umls_to_ontology_with_mentions(
        source_path=paths.st21pv_dir / "all_to_umls_w_mention.json",
        mapping_path=paths.st21pv_dir / "umls2go.json",
        id_list_path=paths.go_dir / "meta" / "biological_process_concept_id_list.json",
        output_path=paths.go_dir / "ori" / "all_w_mention.json",
        ontology_key="go_id",
    )
    collapse_mentions_to_concepts(
        source_path=paths.go_dir / "ori" / "all_w_mention.json",
        ontology_path=paths.go_dir / "meta" / "biological_process_concept.json",
        id_list_path=paths.go_dir / "meta" / "biological_process_concept_id_list.json",
        output_path=paths.go_dir / "ori" / "all_wo_mention.json",
        ontology_key="go_id",
    )

    map_umls_to_ontology_with_mentions(
        source_path=paths.st21pv_dir / "all_to_umls_w_mention.json",
        mapping_path=paths.st21pv_dir / "umls2hpo.json",
        id_list_path=paths.hpo_dir / "meta" / "phenotypic_abnormality_concept_id_list.json",
        output_path=paths.hpo_dir / "ori" / "all_w_mention.json",
        ontology_key="hpo_id",
    )
    collapse_mentions_to_concepts(
        source_path=paths.hpo_dir / "ori" / "all_w_mention.json",
        ontology_path=paths.hpo_dir / "meta" / "phenotypic_abnormality_concept.json",
        id_list_path=paths.hpo_dir / "meta" / "phenotypic_abnormality_concept_id_list.json",
        output_path=paths.hpo_dir / "ori" / "all_wo_mention.json",
        ontology_key="hpo_id",
    )

    build_split_lists(
        data_name="go",
        data_root=paths.data_root,
        output_path=paths.go_dir / "ori" / "split_list.json",
    )
    build_split_lists(
        data_name="hpo",
        data_root=paths.data_root,
        output_path=paths.hpo_dir / "ori" / "split_list.json",
    )

    compute_unseen_statistics("go", paths.data_root)
    compute_unseen_statistics("hpo", paths.data_root)

    export_bm25_jsonl("go", paths.data_root)
    export_bm25_jsonl("hpo", paths.data_root)


if __name__ == "__main__":
    main()
