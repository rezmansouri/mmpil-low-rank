'''empty'''
import os
import h5py
import torch
import numpy as np
import torch.utils
from tqdm import tqdm
from PIL import Image

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
    X, Y = [], []
    for path in paths:
        for h5_name in tqdm(os.listdir(f'{path}'), leave=False):
            with h5py.File(f'{path}/{h5_name}') as h5:
                pil = h5.get('convexhull')[:]
                X.append(pad_and_resize(pil))
                class_ = h5.get('class')[()]
                y = 1 if class_ in [b'X', b'M'] else 0
                Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, modality):
        self.modality = np.expand_dims(modality, axis=1)

    def __len__(self):
        return len(self.modality)

    def __getitem__(self, idx):
        modality = self.modality[idx]
        return torch.tensor(modality, dtype=torch.float32)
