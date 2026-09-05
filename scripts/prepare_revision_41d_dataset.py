#!/usr/bin/env python3
"""Create the frozen 41-D revision dataset from the existing normalized 45-D arrays.

The four removed channels (source indices 38--41) are deterministic actuator
proxies. All remaining normalized values are copied unchanged, preserving the
original train/validation/test split and normalization fitted on training data.
"""

from __future__ import annotations

import argparse
import copy
import pickle
import shutil
from pathlib import Path

import numpy as np


REMOVED = (38, 39, 40, 41)
KEPT = tuple(i for i in range(45) if i not in REMOVED)


def slice_scaler(scaler: object) -> object:
    sliced = copy.deepcopy(scaler)
    for name in ("mean_", "scale_", "var_"):
        value = getattr(sliced, name, None)
        if value is not None:
            setattr(sliced, name, np.asarray(value)[list(KEPT)].copy())
    if hasattr(sliced, "n_features_in_"):
        sliced.n_features_in_ = len(KEPT)
    return sliced


def remap_feature_idx(feature_idx: dict[str, int]) -> dict[str, int]:
    old_to_new = {old: new for new, old in enumerate(KEPT)}
    return {
        name: old_to_new[index]
        for name, index in feature_idx.items()
        if index in old_to_new
    }


def slice_array(source: Path, target: Path, chunk_size: int) -> None:
    array = np.load(source, mmap_mode="r")
    if array.ndim != 3 or array.shape[-1] != 45:
        raise ValueError(f"{source} has shape {array.shape}; expected [N,T,45]")
    output = np.lib.format.open_memmap(
        target,
        mode="w+",
        dtype=array.dtype,
        shape=(*array.shape[:-1], len(KEPT)),
    )
    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])
        output[start:stop] = array[start:stop, :, list(KEPT)]
    output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/dataset_new_processed")
    parser.add_argument("--target", default="data/dataset_revision_41d")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    target = Path(args.target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() and any(target.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{target} is non-empty; pass --overwrite")
    target.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test_id", "test_ood"):
        slice_array(
            source / f"X_{split}.npy",
            target / f"X_{split}.npy",
            args.chunk_size,
        )

    for path in source.iterdir():
        if path.name.startswith("X_") or path.name == "norm_params.pkl":
            continue
        if path.is_file():
            shutil.copy2(path, target / path.name)

    with (source / "norm_params.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    metadata = copy.deepcopy(metadata)
    metadata["scaler_X"] = slice_scaler(metadata["scaler_X"])
    if "feature_names" in metadata:
        metadata["feature_names"] = [
            metadata["feature_names"][index] for index in KEPT
        ]
    if "feature_idx" in metadata:
        metadata["feature_idx"] = remap_feature_idx(metadata["feature_idx"])
    metadata["input_size"] = len(KEPT)
    metadata["source_input_size"] = 45
    metadata["removed_feature_indices"] = list(REMOVED)
    metadata["revision_protocol"] = "config/revision_protocol.yaml"
    with (target / "norm_params.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)

    for split in ("train", "val", "test_id", "test_ood"):
        x = np.load(target / f"X_{split}.npy", mmap_mode="r")
        y = np.load(target / f"y_{split}.npy", mmap_mode="r")
        if x.shape[0] != y.shape[0] or x.shape[-1] != 41:
            raise AssertionError(f"invalid {split}: X={x.shape}, y={y.shape}")
        print(f"{split}: X={x.shape}, y={y.shape}")
    print(f"Wrote frozen 41-D dataset to {target}")


if __name__ == "__main__":
    main()
