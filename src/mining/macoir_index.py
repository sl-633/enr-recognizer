#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ontology concept indexing pipeline for MA-COIR.

Steps:
1) Load ontology names/synonyms (optionally filtered by a concept ID list).
2) Encode texts with a PLM (mean pooling + L2 normalization).
3) Build FAISS (L2) index and semantic similarity graph.
4) (Optional) Add ontology-structure edges.
5) Hierarchical clustering (Louvain + optional METIS fallback).
6) Export embeddings, graphs, and ID→search_id mappings.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import deque
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

try:
    import pymetis

    HAS_PYMETIS = True
except Exception:
    HAS_PYMETIS = False


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    import torch

    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_concept_id_set(concept_id_file: Optional[Path]) -> Optional[Set[str]]:
    if not concept_id_file:
        return None
    obj = load_json(concept_id_file)
    if isinstance(obj, list):
        ids: Iterable[str] = obj
    elif isinstance(obj, dict) and "ids" in obj and isinstance(obj["ids"], list):
        ids = obj["ids"]
    elif isinstance(obj, dict):
        ids = obj.keys()
    else:
        ids = []
    return set(str(i) for i in ids)


def load_target_concepts(ontology_file: Path, concept_id_file: Optional[Path]) -> Dict[str, str]:
    onto = load_json(ontology_file)
    keep_ids = load_concept_id_set(concept_id_file)

    def iter_entries(obj: Any):
        if isinstance(obj, dict):
            for cid, val in obj.items():
                yield str(cid), val
        elif isinstance(obj, list):
            for val in obj:
                cid = val.get("id") or val.get("concept_id") or val.get("obo_id") or val.get("curie")
                if cid:
                    yield str(cid), val

    name_map: Dict[str, str] = {}
    for cid, ent in iter_entries(onto):
        if keep_ids and cid not in keep_ids:
            continue
        name = ent.get("name") or ent.get("label") or ent.get("preferred_label")
        syns = ent.get("synonyms") or ent.get("exact_synonyms") or []
        syns = [s for s in syns if isinstance(s, str)]
        texts = [t for t in [name, *syns] if isinstance(t, str) and t.strip()]
        for idx, text in enumerate(texts):
            name_map[f"{cid}/{idx}"] = text
    return name_map


def encode_texts(
    model: "AutoModel",
    tokenizer: "AutoTokenizer",
    texts: List[str],
    device: "torch.device",
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    vecs = []
    model.eval()
    model.to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state
        pooled = mean_pool(out, enc["attention_mask"])
        pooled = F.normalize(pooled, dim=-1)
        vecs.append(pooled.detach().cpu().numpy())
    if not vecs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def save_embeddings(output_path: Path, idx_list: List[str], embeddings: np.ndarray, labels: List[str]) -> None:
    ensure_dir(output_path)
    with output_path.open("wb") as handle:
        pickle.dump([idx_list, embeddings, labels], handle)


def build_faiss_index_l2(vectors: np.ndarray, use_gpu: bool, gpu: int):
    import faiss

    dim = vectors.shape[1]
    cpu_index = faiss.IndexFlatL2(dim)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu, cpu_index)
        index.add(vectors)
        return index
    cpu_index.add(vectors)
    return cpu_index


def faiss_topk(index, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(queries, topk)


def acquire_neighbors_diverse_l2(
    node: str,
    candidates: List[str],
    max_neighbors: int,
    graph: nx.Graph,
    eps: float = 1e-9,
) -> Tuple[List[str], List[float]]:
    x_vec = np.asarray(graph.nodes[node]["vector"])
    c_vecs = {c: np.asarray(graph.nodes[c]["vector"]) for c in candidates}
    d_to_x = {c: float(np.linalg.norm(x_vec - v)) for c, v in c_vecs.items()}
    order = sorted(candidates, key=lambda c: d_to_x[c])

    neighbors: List[str] = []
    scores: List[float] = []
    for cand in order:
        if len(neighbors) >= max_neighbors:
            break
        d_cx = d_to_x[cand]
        v_c = c_vecs[cand]
        ok = True
        for n in neighbors:
            v_n = c_vecs[n]
            if np.linalg.norm(v_c - v_n) <= d_cx + eps:
                ok = False
                break
        if ok:
            neighbors.append(cand)
            scores.append(d_cx)
    return neighbors, scores


def build_graph(
    concept_embedding_file: Path,
    concept_id_file: Path,
    ontology_file: Path,
    output_graph_path: Path,
    use_semantic_similarity: bool,
    use_ontology_structure: bool,
    l_neighbors: int,
    max_neighbors: int,
    use_gpu_faiss: bool,
    gpu: int,
) -> nx.DiGraph:
    with concept_embedding_file.open("rb") as handle:
        ids, vectors, labels = pickle.load(handle)
    ids = list(ids)
    vectors = np.asarray(vectors)
    labels = list(labels)

    keep_ids = load_concept_id_set(concept_id_file) or set()
    graph = nx.DiGraph()
    kept_nodes = 0

    for idx, vec, label in zip(ids, vectors, labels):
        parts = idx.split("/")
        if len(parts) < 2:
            continue
        root_id = "/".join(parts[:-1])
        tail = parts[-1]
        if root_id in keep_ids and tail == "0":
            node_id = f"C_{kept_nodes}"
            kept_nodes += 1
            graph.add_node(node_id, onto_id=root_id.split("/")[-1], label=str(label), vector=vec.astype(np.float32))

    ontoid_to_nodeid = {graph.nodes[n]["onto_id"]: n for n in graph.nodes}

    if use_semantic_similarity and len(graph.nodes) > 0:
        vecs = np.vstack([graph.nodes[n]["vector"] for n in graph.nodes])
        index = build_faiss_index_l2(vecs, use_gpu=use_gpu_faiss, gpu=gpu)
        node_list = list(graph.nodes)
        added = 0
        for i, n in enumerate(tqdm(node_list, desc="Semantic edges")):
            q = vecs[i : i + 1]
            distances, indices = faiss_topk(index, q, l_neighbors + 1)
            neighbors = [node_list[j] for j in indices[0] if node_list[j] != n]
            selected, dists = acquire_neighbors_diverse_l2(n, neighbors, max_neighbors, graph)
            for tgt, dist in zip(selected, dists):
                graph.add_edge(n, tgt, weight=float(dist) / 2.0)
                added += 1
        print(f"[Semantic] Added {added} edges.")

    if use_ontology_structure:
        onto = load_json(ontology_file)
        added = 0
        if isinstance(onto, dict) and "graphs" in onto:
            for edge in onto["graphs"][0].get("edges", []):
                src = edge.get("obj", "").split("/")[-1]
                tgt = edge.get("sub", "").split("/")[-1]
                if src in ontoid_to_nodeid and tgt in ontoid_to_nodeid:
                    graph.add_edge(ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], weight=0.6)
                    added += 1
        elif isinstance(onto, list):
            for ent in onto:
                tgt = str(ent.get("Class ID", "")).split("/")[-1]
                parents = ent.get("Parents") or []
                for parent in parents:
                    src = str(parent).split("/")[-1]
                    if src in ontoid_to_nodeid and tgt in ontoid_to_nodeid:
                        graph.add_edge(ontoid_to_nodeid[src], ontoid_to_nodeid[tgt], weight=0.6)
                        added += 1
        print(f"[Ontology] Added {added} edges.")

    ensure_dir(output_graph_path)
    with output_graph_path.open("wb") as handle:
        pickle.dump(graph, handle)
    return graph


def dynamic_resolution(
    graph: nx.Graph,
    target_set: Optional[Set[int]] = None,
    search_range: Tuple[float, float] = (1e-6, 5.0),
    max_iters: int = 30,
    seed: int = 666,
) -> List[set]:
    undirected = graph.to_undirected()
    target_set = target_set or {9, 10}
    left, right = search_range
    best_parts, best_diff = None, float("inf")

    for _ in range(max_iters):
        mid = (left + right) / 2
        parts = nx.community.louvain_communities(undirected, resolution=mid, seed=seed)
        k = len(parts)
        diff = min(abs(k - target) for target in target_set)
        if diff < best_diff:
            best_diff = diff
            best_parts = parts
        if k in target_set:
            return list(parts)
        if k > max(target_set):
            right = mid
        else:
            left = mid
    return list(best_parts)


def partition_graph_worker(adj_list: List[List[int]], num_partitions: int, queue: Queue) -> None:
    cuts, membership = pymetis.part_graph(num_partitions, adjacency=adj_list)
    queue.put((cuts, membership))


def perform_partitioning_in_subprocess(G_sub: nx.Graph, num_partitions: int):
    if not HAS_PYMETIS:
        raise RuntimeError("pymetis is not available. Install it to use METIS partitioning.")

    undirected = G_sub.to_undirected()
    mapping = {node: idx for idx, node in enumerate(undirected.nodes())}
    relabeled = nx.relabel_nodes(undirected, mapping, copy=True)
    n = relabeled.number_of_nodes()
    if n == 0:
        return 0, [], {}
    k = min(num_partitions, max(1, n))
    if k <= 1:
        idx_to_orig = {idx: node for node, idx in mapping.items()}
        return 0, [0] * n, idx_to_orig

    adj_list = [list(relabeled.neighbors(i)) for i in range(n)]
    queue = Queue()
    process = Process(target=partition_graph_worker, args=(adj_list, k, queue))
    process.start()
    process.join()
    if queue.empty():
        raise RuntimeError("PyMetis process returned no result.")
    cuts, membership = queue.get()
    idx_to_orig = {idx: node for node, idx in mapping.items()}
    return cuts, membership, idx_to_orig


def top_level_partition(
    graph: nx.DiGraph,
    strict_top_range: Tuple[int, int],
    seed: int,
    use_metis_on_top_fail: bool,
    metis_top_k: int,
) -> List[set]:
    target = set(range(strict_top_range[0], strict_top_range[1] + 1))
    parts = dynamic_resolution(graph, target_set=target, seed=seed)
    k = len(parts)
    if strict_top_range[0] <= k <= strict_top_range[1]:
        return parts
    if not use_metis_on_top_fail:
        raise RuntimeError(f"Top-level Louvain got k={k}, outside {strict_top_range}.")

    cuts, membership, idx_to_orig = perform_partitioning_in_subprocess(graph, metis_top_k)
    uniq = sorted(set(membership))
    buckets = []
    for cid in uniq:
        nodes_c = {idx_to_orig[i] for i, m in enumerate(membership) if m == cid}
        buckets.append(nodes_c)
    return buckets


def save_checkpoint(checkpoint_dir: Path, graph: nx.Graph, pending_list: List[str]) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    (checkpoint_dir / "pending.json").write_text(json.dumps(pending_list, ensure_ascii=False, indent=2), encoding="utf-8")
    status = f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\nremaining={len(pending_list)}\n"
    (checkpoint_dir / "status.txt").write_text(status, encoding="utf-8")


def load_checkpoint(checkpoint_dir: Path) -> Tuple[nx.Graph, List[str]]:
    with (checkpoint_dir / "graph.pkl").open("rb") as handle:
        graph = pickle.load(handle)
    pending = json.loads((checkpoint_dir / "pending.json").read_text(encoding="utf-8"))
    return graph, pending


def louvain_phase_with_top_hybrid(
    graph_in: nx.DiGraph,
    target_per_leaf: int,
    strict_top_range: Tuple[int, int],
    use_metis_on_top_fail: bool,
    metis_top_k: int,
) -> Tuple[nx.DiGraph, List[str]]:
    graph = nx.DiGraph()
    graph.update(graph_in)
    pending: List[str] = []

    top_parts = top_level_partition(
        graph,
        strict_top_range=strict_top_range,
        seed=666,
        use_metis_on_top_fail=use_metis_on_top_fail,
        metis_top_k=metis_top_k,
    )

    for i, nodes in enumerate(top_parts):
        for n in nodes:
            graph.nodes[n]["cluster"] = f"{i}"

    def rec(cluster_id: str) -> None:
        nodes = [n for n in graph.nodes if graph.nodes[n]["cluster"] == cluster_id]
        m = len(nodes)
        if m <= 1:
            return
        if 1 < m <= target_per_leaf:
            for j, n in enumerate(nodes):
                graph.nodes[n]["cluster"] = f"{cluster_id}-{j}"
            return

        sub = graph.subgraph(nodes).copy()
        parts = dynamic_resolution(sub, target_set={8, 9, 10})
        k = len(parts)
        if k == 1:
            return
        if k <= target_per_leaf:
            for j, ns in enumerate(parts):
                for n in ns:
                    graph.nodes[n]["cluster"] = f"{cluster_id}-{j}"
            for j in range(k):
                rec(f"{cluster_id}-{j}")
        else:
            pending.append(cluster_id)

    for i in range(len(top_parts)):
        rec(str(i))

    pending = sorted(set(pending))
    return graph, pending


def metis_phase_iterative_with_checkpoints(
    graph: nx.DiGraph,
    pending_clusters: List[str],
    target_per_leaf: int,
    checkpoint_dir: Path,
    save_every: int,
    recursive_metis: bool,
) -> nx.DiGraph:
    queue = deque(pending_clusters)
    processed = 0
    total = len(queue)

    while queue:
        cluster_id = queue.popleft()
        processed += 1
        nodes = [n for n in graph.nodes if graph.nodes[n]["cluster"] == cluster_id]
        m = len(nodes)
        if m <= 1:
            if processed % max(1, save_every) == 0:
                save_checkpoint(checkpoint_dir, graph, list(queue))
            continue
        if 1 < m <= target_per_leaf:
            for j, n in enumerate(nodes):
                graph.nodes[n]["cluster"] = f"{cluster_id}-{j}"
            if processed % max(1, save_every) == 0:
                save_checkpoint(checkpoint_dir, graph, list(queue))
            continue

        sub = graph.subgraph(nodes).copy()
        try:
            _, membership, idx_to_orig = perform_partitioning_in_subprocess(sub, target_per_leaf)
        except Exception as exc:
            print(f"[METIS][ERROR] {cluster_id}: {exc}. Will retry later.")
            queue.append(cluster_id)
            save_checkpoint(checkpoint_dir, graph, list(queue))
            continue

        uniq = sorted(set(membership))
        if len(uniq) <= 1:
            if processed % max(1, save_every) == 0:
                save_checkpoint(checkpoint_dir, graph, list(queue))
            continue

        for i, cid in enumerate(membership):
            node = idx_to_orig[i]
            graph.nodes[node]["cluster"] = f"{cluster_id}-{cid}"

        if recursive_metis:
            child_counts: Dict[int, int] = {}
            for cid in membership:
                child_counts[cid] = child_counts.get(cid, 0) + 1
            for cid, cnt in child_counts.items():
                child_id = f"{cluster_id}-{cid}"
                if cnt <= 1:
                    continue
                if cnt <= target_per_leaf:
                    child_nodes = [idx_to_orig[i] for i, m_c in enumerate(membership) if m_c == cid]
                    for j, n in enumerate(child_nodes):
                        graph.nodes[n]["cluster"] = f"{child_id}-{j}"
                else:
                    queue.append(child_id)

        if processed % max(1, save_every) == 0:
            save_checkpoint(checkpoint_dir, graph, list(queue))

    save_checkpoint(checkpoint_dir, graph, [])
    return graph


def hierarchical_partition(
    graph_in: nx.DiGraph,
    target_per_leaf: int,
    checkpoint_dir: Path,
    resume_from_checkpoint: bool,
    save_every: int,
    strict_top_range: Tuple[int, int],
    use_metis_on_top_fail: bool,
    metis_top_k: int,
) -> nx.DiGraph:
    ckpt_graph = checkpoint_dir / "graph.pkl"
    ckpt_pending = checkpoint_dir / "pending.json"

    if resume_from_checkpoint and ckpt_graph.exists() and ckpt_pending.exists():
        graph, pending = load_checkpoint(checkpoint_dir)
    else:
        graph, pending = louvain_phase_with_top_hybrid(
            graph_in,
            target_per_leaf=target_per_leaf,
            strict_top_range=strict_top_range,
            use_metis_on_top_fail=use_metis_on_top_fail,
            metis_top_k=metis_top_k,
        )
        save_checkpoint(checkpoint_dir, graph, pending)

    if pending:
        graph = metis_phase_iterative_with_checkpoints(
            graph,
            pending_clusters=pending,
            target_per_leaf=target_per_leaf,
            checkpoint_dir=checkpoint_dir,
            save_every=save_every,
            recursive_metis=True,
        )
    return graph


def map_index(graph_path: Path, csv_path: Path, indexing_path: Path, do_random: bool) -> None:
    with graph_path.open("rb") as handle:
        graph: nx.DiGraph = pickle.load(handle)
    graph_copy = nx.DiGraph()
    graph_copy.update(graph)
    graph = graph_copy

    id_to_cluster: Dict[str, List[int]] = {}
    labels: List[str] = []
    for n in graph.nodes:
        onto_id = str(graph.nodes[n]["onto_id"])
        label = str(graph.nodes[n]["label"])
        if not onto_id.startswith(("GO_", "HP_")):
            continue
        cluster_path = [int(x) for x in str(graph.nodes[n].get("cluster", "0")).split("-")]
        id_to_cluster[onto_id] = cluster_path
        labels.append(label)

    concept_ids = list(id_to_cluster.keys())
    cluster_ids = [id_to_cluster[cid] for cid in concept_ids]
    if do_random:
        random.shuffle(cluster_ids)
        id_to_cluster = {cid: sid for cid, sid in zip(concept_ids, cluster_ids)}

    ensure_dir(csv_path)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("concept_id\tsearch_id\tlabel\n")
        for concept_id, cluster_id, label in zip(concept_ids, cluster_ids, labels):
            handle.write(f"{concept_id}\t{cluster_id}\t{label}\n")

    ensure_dir(indexing_path)
    indexing_path.write_text(json.dumps(id_to_cluster, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontology concept indexing pipeline")
    parser.add_argument("--ontology_file", type=Path, required=True)
    parser.add_argument("--concept_file", type=Path, required=True)
    parser.add_argument("--concept_id_file", type=Path, required=True)

    parser.add_argument("--model_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--embed_max_len", type=int, default=128)
    parser.add_argument("--embed_bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--embedding_pickle", type=Path, required=True)
    parser.add_argument("--graph_pickle", type=Path, required=True)
    parser.add_argument("--partitioned_graph_pickle", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--indexing_path", type=Path, required=True)

    parser.add_argument("--use_semantic_similarity", action="store_true")
    parser.add_argument("--use_ontology_structure", action="store_true")
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--M", type=int, default=10)

    parser.add_argument("--target_leaf_size", type=int, default=10)
    parser.add_argument("--save_step", type=int, default=1)

    parser.add_argument("--skip_embed", action="store_true")
    parser.add_argument("--skip_build_graph", action="store_true")
    parser.add_argument("--resume_partition", action="store_true")
    parser.add_argument("--metis_top_k", type=int, default=10)
    parser.add_argument("--no_metis_fallback", action="store_true")
    parser.add_argument("--strict_top_min", type=int, default=8)
    parser.add_argument("--strict_top_max", type=int, default=10)
    parser.add_argument("--randomize_clusters", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ontology_file.exists():
        raise SystemExit(f"--ontology_file not found: {args.ontology_file}")
    if not args.concept_file.exists():
        raise SystemExit(f"--concept_file not found: {args.concept_file}")
    if not args.concept_id_file.exists():
        raise SystemExit(f"--concept_id_file not found: {args.concept_id_file}")

    import torch
    from transformers import AutoModel, AutoTokenizer

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.skip_embed and args.embedding_pickle.exists():
        with args.embedding_pickle.open("rb") as handle:
            ids, vectors, labels = pickle.load(handle)
    else:
        name_map = load_target_concepts(args.concept_file, args.concept_id_file)
        ids = list(name_map.keys())
        texts = [name_map[k] for k in ids]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModel.from_pretrained(args.model_name)
        vectors = encode_texts(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.embed_max_len,
            batch_size=args.embed_bs,
        )
        labels = texts[:]
        save_embeddings(args.embedding_pickle, ids, vectors, labels)

    if args.skip_build_graph and args.graph_pickle.exists():
        with args.graph_pickle.open("rb") as handle:
            graph = pickle.load(handle)
    else:
        graph = build_graph(
            concept_embedding_file=args.embedding_pickle,
            concept_id_file=args.concept_id_file,
            ontology_file=args.ontology_file,
            output_graph_path=args.graph_pickle,
            use_semantic_similarity=args.use_semantic_similarity,
            use_ontology_structure=args.use_ontology_structure,
            l_neighbors=args.L,
            max_neighbors=args.M,
            use_gpu_faiss=args.faiss_gpu,
            gpu=args.gpu,
        )

    graph_partitioned = hierarchical_partition(
        graph_in=graph,
        target_per_leaf=args.target_leaf_size,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_partition,
        save_every=args.save_step,
        strict_top_range=(args.strict_top_min, args.strict_top_max),
        use_metis_on_top_fail=not args.no_metis_fallback,
        metis_top_k=args.metis_top_k,
    )

    ensure_dir(args.partitioned_graph_pickle)
    with args.partitioned_graph_pickle.open("wb") as handle:
        pickle.dump(graph_partitioned, handle)

    map_index(
        graph_path=args.partitioned_graph_pickle,
        csv_path=args.csv_path,
        indexing_path=args.indexing_path,
        do_random=args.randomize_clusters,
    )


if __name__ == "__main__":
    main()
