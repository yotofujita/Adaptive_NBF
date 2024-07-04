import json
import numpy as np
from numpy.random import Generator, PCG64


def read_manifest(path):
    with open(path, "r") as f:
        manifest = f.readlines()
    manifest = [json.loads(m) for m in manifest]
    return manifest


def write_manifest(manifest, path, ensure_ascii=False):
    with open(path, "w") as f:
        for m in manifest:
            json.dump(m, f, ensure_ascii=ensure_ascii)
            f.write("\n")


def scatter_indices(dataset_size, rank, world_size, permute_fn=Generator(PCG64(0)).permutation):
    total_size = int(np.ceil(dataset_size / world_size)) * world_size

    indices = permute_fn(np.arange(dataset_size))
    repeated_indices = np.concatenate([indices, indices[:total_size - dataset_size]])

    split_indices = np.split(repeated_indices, world_size)

    return split_indices[rank]