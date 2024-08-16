'''empty'''
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm


def get_raw_data(paths):
    '''empty'''
    print(f'reading from {paths}')
    X, Y = [], []
    for path in paths:
        for h5_name in tqdm(os.listdir(f'{path}'), leave=False):
            with h5py.File(f'{path}/{h5_name}') as h5:
                properties = h5.get('properties')[:]
                X.append([
                    [float(item[1].decode()) if item[1].decode() != '' else np.NaN for item in p] for p in properties
                ])
                class_ = h5.get('class')[()]
                y = 1 if class_ in [b'X', b'M'] else 0
                Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def interpolate_properties(X):
    '''empty'''
    for ix, x in enumerate(X):
        if not np.isnan(x).any():
            continue
        needle = 0
        while needle < len(x):
            t = x[needle]
            if np.isnan(t).any():
                prev_t = x[needle-1]
                next_t = None
                n_missing = 1
                for temp_next_t in x[needle+1:]:
                    if not np.isnan(temp_next_t).any():
                        next_t = temp_next_t
                        break
                    else:
                        n_missing += 1
                diff = next_t - prev_t
                step = diff / (n_missing + 1)
                for i in range(needle, needle+n_missing):
                    X[ix, i, :] = X[ix, i-1, :] + step
                needle += n_missing + 1
                continue
            needle += 1
    return X


def z_normalization(X):
    '''empty'''
    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    X = (X - mean) / std
    X = np.asarray(X, dtype=np.float32)
    return X, mean, std


class CustomDataset(torch.utils.data.Dataset):
    '''empty'''

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float)
        return sample


def batch_second(batch):
    '''empty'''
    return torch.stack(batch, dim=1)
