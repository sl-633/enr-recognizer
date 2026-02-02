"""Preprocessing utilities for ontology datasets."""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Iterable

import networkx as nx

BASE_IRI = "http://purl.obolibrary.org/obo/"
HPO_ROOT = f"{BASE_IRI}HP_0000118"
GO_ROOT = f"{BASE_IRI}GO_0008150"

logger = logging.getLogger(__name__)


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict | list) -> None:
    """Write a JSON file to disk, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_concept_graph(ontology_path: Path, require_label: bool = True) -> tuple[nx.DiGraph, dict[str, str]]:
    """Build a directed graph from an ontology JSON file.

    Returns the graph and a mapping from node IDs in the graph to ontology IRIs.
    """
    ontology = load_json(ontology_path)["graphs"][0]
    nodes = ontology.get("nodes", [])
    edges = ontology.get("edges", [])

    concept_ids: list[str] = []
    labels: list[str] = []
    for node in nodes:
        if node.get("type") != "CLASS":
            continue
        if require_label and not node.get("lbl"):
            continue
        concept_ids.append(node["id"])
        labels.append(node.get("lbl", ""))

    concept_info = list(zip(concept_ids, labels, strict=True))

    nodeid_to_ontoid = {f"name_{idx}": onto_id for idx, (onto_id, _) in enumerate(concept_info)}
    ontoid_to_nodeid = {onto_id: node_id for node_id, onto_id in nodeid_to_ontoid.items()}

    graph = nx.DiGraph()
    graph.add_nodes_from(
        (f"name_{idx}", {"onto_id": onto_id, "text": label})
        for idx, (onto_id, label) in enumerate(concept_info)
    )

    edge_set: set[tuple[str, str]] = set()
    for relation in edges:
        obj_id = ontoid_to_nodeid.get(relation.get("obj"))
        sub_id = ontoid_to_nodeid.get(relation.get("sub"))
        if not obj_id or not sub_id:
            continue
        edge_set.add((obj_id, sub_id))

    graph.add_edges_from(edge_set)

    logger.info("Loaded %s concepts and %s edges from %s", len(graph.nodes), len(graph.edges), ontology_path)
    return graph, nodeid_to_ontoid


def get_all_descendants(graph: nx.DiGraph, start_node: str) -> list[str]:
    """Return all descendants of a node using breadth-first search."""
    visited: set[str] = set()
    queue: deque[str] = deque([start_node])
    while queue:
        current = queue.popleft()
        for neighbor in graph.successors(current):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return list(visited)


def extract_descendant_ids(
    ontology_path: Path,
    root_iri: str,
    output_path: Path,
    require_label: bool = True,
) -> list[str]:
    """Extract descendant ontology IDs from a root term and save them."""
    graph, nodeid_to_ontoid = build_concept_graph(ontology_path, require_label=require_label)
    ontoid_to_nodeid = {onto_id: node_id for node_id, onto_id in nodeid_to_ontoid.items()}

    start_node = ontoid_to_nodeid[root_iri]
    descendants = get_all_descendants(graph, start_node)
    descendant_ids = [nodeid_to_ontoid[node_id].replace(BASE_IRI, "") for node_id in descendants]

    logger.info("Root %s has %s descendants", root_iri, len(descendants))
    write_json(output_path, descendant_ids)
    return descendant_ids


def get_name_and_synonyms(ontology_path: Path, id_list_path: Path, output_path: Path) -> None:
    """Filter ontology concepts by ID list and export names/synonyms."""
    ontology = load_json(ontology_path)["graphs"][0]
    nodes = ontology.get("nodes", [])

    id_list = set(load_json(id_list_path))

    concept_set_info: list[tuple[str, str, list[str]]] = []
    for node in nodes:
        if node.get("type") != "CLASS" or not node.get("lbl"):
            continue
        compact_id = node["id"].replace(BASE_IRI, "")
        if compact_id not in id_list:
            continue

        synonyms = [synonym["val"] for synonym in node.get("meta", {}).get("synonyms", [])]
        concept_set_info.append((compact_id, node["lbl"], synonyms))

    logger.info("Number of target concepts: %s", len(concept_set_info))
    logger.info(
        "Number of synonyms of target concepts: %s",
        sum(len(synonyms) for _, _, synonyms in concept_set_info),
    )

    payload = {concept_id: {"name": name, "synonyms": synonyms} for concept_id, name, synonyms in concept_set_info}
    write_json(output_path, payload)


def get_onto_jsonl(input_path: Path, output_path: Path) -> None:
    """Convert ontology concepts to JSONL for BM25 mining."""
    onto_info = load_json(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for concept_id, record in onto_info.items():
            payload = {"id": concept_id, "name": record["name"], "synonyms": record["synonyms"]}
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")

    logger.info("Finished transformation, output: %s", output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ontology datasets for mining and training.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../../data"),
        help="Root directory containing mm-go and mm-hpo datasets.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()
    data_root: Path = args.data_root

    hpo_meta = data_root / "mm-hpo" / "meta"
    go_meta = data_root / "mm-go" / "meta"

    hpo_descendants = extract_descendant_ids(
        ontology_path=hpo_meta / "hp.json",
        root_iri=HPO_ROOT,
        output_path=hpo_meta / "phenotypic_abnormality_concept_id_list.json",
        require_label=True,
    )

    go_descendants = extract_descendant_ids(
        ontology_path=go_meta / "go-basic.json",
        root_iri=GO_ROOT,
        output_path=go_meta / "biological_process_concept_id_list.json",
        require_label=True,
    )

    logger.info("Extracted %s HPO descendants and %s GO descendants", len(hpo_descendants), len(go_descendants))

    get_name_and_synonyms(
        ontology_path=go_meta / "go-basic.json",
        id_list_path=go_meta / "biological_process_concept_id_list.json",
        output_path=go_meta / "biological_process_concept.json",
    )
    get_name_and_synonyms(
        ontology_path=hpo_meta / "hp.json",
        id_list_path=hpo_meta / "phenotypic_abnormality_concept_id_list.json",
        output_path=hpo_meta / "phenotypic_abnormality_concept.json",
    )

    get_onto_jsonl(
        input_path=go_meta / "biological_process_concept.json",
        output_path=go_meta / "onto_for_bm25.jsonl",
    )
    get_onto_jsonl(
        input_path=hpo_meta / "phenotypic_abnormality_concept.json",
        output_path=hpo_meta / "onto_for_bm25.jsonl",
    )


if __name__ == "__main__":
    main()
