#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare XR-Transformer inputs and postprocess XR-Transformer predictions."""

from __future__ import annotations

import argparse
import json
import logging
import os.path
import numpy as np
import scipy.sparse as smat
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, save_npz

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# Copied from `pecos.pecos.utils.smat_util`
def load_matrix(src, dtype=None):
    """Load dense or sparse matrix from file.

    Args:
        src (str): path to load the matrix.
        dtype (numpy.dtype, optional): if given, convert matrix dtype. otherwise use default type.

    Returns:
        mat (numpy.ndarray or scipy.sparse.spmatrix): loaded matrix

    Notes:
        If underlying matrix is {"csc", "csr", "bsr"}, indices will be sorted.
    """
    if not isinstance(src, str):
        raise ValueError("src for load_matrix must be a str")

    mat = np.load(src)
    # decide whether it's dense or sparse
    if isinstance(mat, np.ndarray):
        pass
    elif isinstance(mat, np.lib.npyio.NpzFile):
        # Ref code: https://github.com/scipy/scipy/blob/v1.4.1/scipy/sparse/_matrix_io.py#L19-L80
        matrix_format = mat["format"].item()
        if not isinstance(matrix_format, str):
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            matrix_format = matrix_format.decode("ascii")
        try:
            cls = getattr(smat, "{}_matrix".format(matrix_format))
        except AttributeError:
            raise ValueError("Unknown matrix format {}".format(matrix_format))

        if matrix_format in ("csc", "csr", "bsr"):
            mat = cls((mat["data"], mat["indices"], mat["indptr"]), shape=mat["shape"])
            # This is in-place operation
            mat.sort_indices()
        elif matrix_format == "dia":
            mat = cls((mat["data"], mat["offsets"]), shape=mat["shape"])
        elif matrix_format == "coo":
            mat = cls((mat["data"], (mat["row"], mat["col"])), shape=mat["shape"])
        else:
            raise NotImplementedError(
                "Load is not implemented for sparse matrix of format {}.".format(matrix_format)
            )
    else:
        raise TypeError("load_feature_matrix encountered unknown input format {}".format(type(mat)))

    if dtype is None:
        return mat
    else:
        return mat.astype(dtype)


def build_xrt_inputs(data_name: str, data_root: Path, concepts: Path, id_map_out: Path) -> None:
    concept_list = load_json(data_root / f"mm-{data_name}" / "meta" / concepts)
    id_map = {concept: i for i, concept in enumerate(sorted(concept_list))}
    write_json(data_root / f"mm-{data_name}" / "meta" / id_map_out, id_map)
    print(f"Stored the label_id to concept id mapping, output: {id_map_out}")

    all_data = load_json(data_root / f"mm-{data_name}" / "ori" / "all_wo_mention.json")
    split_list = load_json(data_root / f"mm-{data_name}" / "ori" / "split_list.json")

    if not os.path.exists(data_root / f"mm-{data_name}" / "mld" / "xrt-input"):
        os.makedirs(data_root / f"mm-{data_name}" / "mld" / "xrt-input")

    vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', max_features=None)

    def create_label_matrix(arr):
        num_rows = len(arr)
        num_cols = len(concept_list)

        row_indices = []
        col_indices = []
        data = []

        for i, labels in enumerate(arr):
            for label in labels:
                row_indices.append(i)
                col_indices.append(label)
                data.append(1.0)

        label_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))

        return label_matrix

    for split in ["train", "dev", "test"]:
        x_texts_out_path = data_root / f"mm-{data_name}" / "mld" / "xrt-input" / f"X.{split}.txt"
        y_ids_out_path = data_root / f"mm-{data_name}" / "mld" / "xrt-input" / f"Y.{split}.txt"
        x_npz_out_path = data_root / f"mm-{data_name}" / "mld" / "xrt-input" / f"X.{split}.npz"
        y_npz_out_path = data_root / f"mm-{data_name}" / "mld" / "xrt-input" / f"X.{split}.npz"

        x_texts, y_ids, unique_concept_names, name_id_map = [], [], set(), {}
        for doc_id, doc in all_data.items():
            if doc_id not in split_list.get(split, []):
                continue
            query_text = doc["passage"]
            x_texts.append(query_text)
            cur_y_ids = []
            for c in doc["concepts"]:
                unique_concept_names.add(c["name"])
                name_id_map[c["name"]] = id_map[c["id"]]
                cur_y_ids.append(str(id_map[c["id"]]))
            y_ids.append(cur_y_ids)

        if split == "train":
            for name in unique_concept_names:
                x_texts.append(name)
                y_ids.append([name_id_map[name]])

            vectorizer.fit(x_texts)

        with x_texts_out_path.open("w", encoding="utf-8") as out:
            for record in x_texts:
                out.write(record + "\n")
        print(f"Finished transformation, output: {x_texts_out_path}")

        X = vectorizer.transform(x_texts)
        csr_text = csr_matrix(X)

        save_npz(x_npz_out_path, csr_text)
        print(f"Finished text to matrix transformation, output: {x_npz_out_path}")

        with y_ids_out_path.open("w", encoding="utf-8") as out:
            for record in y_ids:
                r = ",".join([str(i) for i in record])
                out.write(str(r) + "\n")
        print(f"Finished transformation, output: {y_ids_out_path}")

        y_ids = create_label_matrix(y_ids)
        save_npz(y_npz_out_path, y_ids)
        print(f"Finished label to matrix transformation, output: {y_npz_out_path}")


def export_xrt_predictions(data_name: str, data_root: Path, id_map_path: Path, id_name_path: Path, model_outputs: Path,
                           topk: int):
    pred_matrix = load_matrix(f"{data_root}/mm-{data_name}/model_outputs/{model_outputs}")
    pred_array = pred_matrix.toarray()
    pred_indice = []

    for i, m in enumerate(pred_array):
        top_ind = np.argpartition(m, -topk)[-topk:]
        top_val = m[np.argpartition(m, -topk)[-topk:]]
        ind_val = [(ind, val) for ind, val in zip(top_ind.tolist(), top_val.tolist())]

        sorted_ind_val = sorted(ind_val, key=lambda x: x[1], reverse=True)
        pred_indice.append(sorted_ind_val)

    id_map = load_json(Path(f"data/mm-{data_name}/meta/{id_map_path}"))
    id_cid_map = {i: cid for cid, i in id_map.items()}
    id_name_map = load_json(Path(f"data/mm-{data_name}/meta/{id_name_path}"))

    with open(f"data/mm-{data_name}/mld/xrt-input/X.{model_outputs}.txt", "r") as f:
        lines = f.readlines()
        gold_texts = []
        for line in lines:
            if len(line.strip()) > 0:
                gold_texts.append(line.strip())
        f.close()

    res = []
    count_d = 0

    for p_ind_val in pred_indice:
        items = []
        for i, e in enumerate(p_ind_val):
            items.append({
                "rank": i,
                "score": e[1],
                "term_text": id_name_map[id_cid_map[e[0]]]["name"],
                "term_id": id_cid_map[e[0]]
            })

        new_res = {
            "query_text": gold_texts[count_d],
            "items": items
        }
        res.append(new_res)
        count_d += 1

    prediction_output = data_root / f"mm-{data_name}" / "top-100-predictions" / f"{model_outputs}.jsonl"
    if not os.path.exists(data_root / f"mm-{data_name}" / "top-100-predictions"):
        os.makedirs(data_root / f"mm-{data_name}" / "top-100-predictions")

    with prediction_output.open("w", encoding="utf-8") as out:
        for r in res:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info("Finished transformation, output: %s", prediction_output)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=Path("logs/xrt"),
                        help="Root directory containing xr-transformer's inputs/outputs of mm-go and mm-hpo datasets.")
    parser.add_argument("--data-name", type=str, required=True,
                        help="Set as go or hpo for specifying the dataset ")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocessing XR-Transformer's inputs or postprocessing its outputs.")

    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="Preprocessing XR-Transformer's inputs.")
    add_shared_args(preprocess)
    preprocess.add_argument("--concepts", type=Path, required=True, help="Path to concept list JSON.")
    preprocess.add_argument("--id_map_out", type=Path,
                            help="Path to mapping between ontological concept ID and xr-transformer's label ID")

    postprocess = sub.add_parser("postprocess", help="Postprocessing XR-Transformer's outputs.")
    add_shared_args(postprocess)
    postprocess.add_argument("--id_map_path", type=Path,
                             help="Path to mapping between ontological concept ID and xr-transformer's label ID")
    postprocess.add_argument("--id_name_path", type=Path, help="Path to ontological concept information")
    postprocess.add_argument("--topk", type=int, default=100, help="Candidate pool size.")
    postprocess.add_argument("--model_outputs", type=Path, default=None, help="Write per-doc predictions JSONL.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    if args.command == "preprocess":
        build_xrt_inputs(args.data_name, args.data_root, args.concepts, args.id_map_out)

    if args.command == "postprocess":
        export_xrt_predictions(args.data_name, args.data_root, args.id_map_path, args.id_name_path, args.model_outputs,
                               args.topk)


if __name__ == "__main__":
    main()
