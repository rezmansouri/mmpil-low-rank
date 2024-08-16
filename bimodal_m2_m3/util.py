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


def collate(batch):
    pils, chs, ys = [], [], []
    for sample in batch:
        pil, ch, y = sample
        pils.append(pil)
        chs.append(ch)
        ys.append(y)
    pils = np.array(pils, dtype=np.float32)
    chs = np.array(chs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    return (torch.tensor(pils, dtype=torch.float32).to(device),
            torch.tensor(chs, dtype=torch.float32).to(device)), torch.tensor(ys, dtype=torch.float32).to(device)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, modalities, labels):
        self.pil, self.ch = modalities
        self.labels = np.expand_dims(labels, axis=1)
        self.pil = np.expand_dims(self.pil, axis=1)
        self.ch = np.expand_dims(self.ch, axis=1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        pil = self.pil[index]
        ch = self.ch[index]
        label = self.labels[index]
        return pil, ch, label
    
    
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
