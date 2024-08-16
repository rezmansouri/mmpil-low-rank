import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_and_resize(image, size=(128, 128)):
    '''empty'''
    h, w = image.shape
    v_pad = max(0, w-h) // 2
    h_pad = max(0, h-w) // 2
    padded = np.pad(image, ((v_pad, v_pad), (h_pad, h_pad)))
    img = Image.fromarray(padded * 255)
    img = img.resize(size, Image.LANCZOS)
    img = np.array(img)
    img[img != 0] = 1
    return img

def get_raw_data(paths):
    '''empty'''
    print(f'reading from {paths}')
    X_properties, X_pil, X_ch, Y = [], [], [], []
    for path in paths:
        for h5_name in tqdm(os.listdir(f'{path}'), leave=False):
            with h5py.File(f'{path}/{h5_name}') as h5:
                properties = h5.get('properties')[:]
                X_properties.append([
                    [float(item[1].decode()) if item[1].decode() != '' else np.NaN for item in p] for p in properties
                ])
                X_pil.append(pad_and_resize(h5.get('pil')[:]))
                X_ch.append(pad_and_resize(h5.get('convexhull')[:]))
                class_ = h5.get('class')[()]
                y = 1 if class_ in [b'X', b'M'] else 0
                Y.append(y)
    return np.array(X_properties, dtype=np.float32), np.array(X_pil, dtype=np.float32), np.array(X_ch, dtype=np.float32), np.array(Y, dtype=np.float32)


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
    X = np.asarray(X, dtype=np.float128)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    X = (X - mean) / std
    X = np.asarray(X, dtype=np.float32)
    return X, mean, std


def collate(batch):
    propertiess, pils, ys = [], [], []
    for sample in batch:
        properties, pil, y = sample
        propertiess.append(properties)
        pils.append(pil)
        ys.append(y)
    propertiess = np.array(propertiess, dtype=np.float32)
    propertiess = np.transpose(propertiess, axes=(1, 0, 2))
    pils = np.array(pils, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    return (torch.tensor(propertiess, dtype=torch.float32).to(device),
            torch.tensor(pils, dtype=torch.float32).to(device)), torch.tensor(ys, dtype=torch.float32).to(device)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, modalities, labels):
        self.properties, self.pil = modalities
        self.labels = np.expand_dims(labels, axis=1)
        self.pil = np.expand_dims(self.pil, axis=1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        properties = self.properties[index]
        pil = self.pil[index]
        label = self.labels[index]
        return properties, pil, label
    
    
class PILSampler(torch.utils.data.Sampler):
    def __init__(self, labels):
        self.labels = labels
        self.n_samples = len(labels)
        self.positive_indices = np.where(labels == 1.)[0]
        self.negative_indices = np.where(labels == 0.)[0]
        self.positive_length = len(self.positive_indices)
        self.negative_length = len(self.negative_indices)
        self.weights = torch.ones(self.n_samples, dtype=torch.float32)
        self.weights[self.positive_indices] /= self.positive_length
        self.weights[self.negative_indices] /= self.negative_length
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.n_samples, replacement=True))
    def __len__(self):
        length = self.negative_length * 2 if self.positive_length < self.negative_length else self.positive_length * 2
        return length
