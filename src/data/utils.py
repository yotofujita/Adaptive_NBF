import h5py

import torch

from multiprocessing import Manager


class NumpiedTensor:
    def __init__(self, tensor):
        self.array = tensor.numpy()

    def to_tensor(self):
        return torch.tensor(self.array)


def numpize_sample(sample):
    if isinstance(sample, torch.Tensor):
        return NumpiedTensor(sample)
    elif isinstance(sample, tuple):
        return tuple(numpize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [numpize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: numpize_sample(v) for k, v in sample.items()}
    else:
        return sample


def tensorize_sample(sample):
    if isinstance(sample, NumpiedTensor):
        return sample.to_tensor()
    elif isinstance(sample, tuple):
        return tuple(tensorize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [tensorize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: tensorize_sample(v) for k, v in sample.items()}
    else:
        return sample


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.manager = Manager()
        self.cache = self.manager.dict()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = numpize_sample(self.dataset[index])

        return tensorize_sample(self.cache[index])


class HDF5RawDataset_old(torch.utils.data.Dataset):
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
        
        self.grp_list = []
        self.fl = []
        for i, dp in enumerate(dataset_paths):
            f = h5py.File(dp, "r")
            self.grp_list += [(i, key) for key in range(f["angle"].shape[0])]
            self.fl.append(f)

    def __len__(self):
        return len(self.grp_list)

    def __getitem__(self, index):
        file_id, idx = self.grp_list[index]
        
        sample = {k: self.fl[file_id][k][idx] for k in self.fl[file_id].keys()}

        return sample, index

    def __del__(self):
        for f in self.fl:
            f.close()


# class HDF5RawDataset_download(torch.utils.data.Dataset):
#     def __init__(self, dataset_path):
    
#         with h5py.File(dataset_path, "r") as f:
#             self.angles = f["angle"]
#             self.images = f["image"]
#             self.mixtures = f["mixture"]

#     def __len__(self):
#         return len(self.angles)

#     def __getitem__(self, index):
#         sample = {
#             "angle": self.angles[index],
#             "image": self.images[index],
#             "mixture": self.mixtures[index]
#         }

#         return sample, index

#     def __del__(self):
#         for f in self.fl:
#             f.close()


class HDF5RawDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        with h5py.File(self.dataset_path, "r") as f:
            self.grp_list = sorted(list(f.keys()))

        self.f = None

    def __len__(self):
        return len(self.grp_list)

    def __getitem__(self, index):
        if self.f is None:
            self.f = h5py.File(self.dataset_path, "r")

        sample = {k: v[:] for k, v in self.f[self.grp_list[index]].items()}

        return sample, index

    def __del__(self):
        if self.f is not None:
            self.f.close()