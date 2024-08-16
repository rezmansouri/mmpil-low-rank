'''empty'''
import os
import sys
import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader, Sampler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-9


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
        for h5_name in tqdm(os.listdir(path), leave=False):
            with h5py.File(os.path.join(path, h5_name), 'r') as h5:
                X.append(pad_and_resize(h5.get('convexhull')[:]))
                class_ = h5.get('class')[()]
                y = 1 if class_ in [b'X', b'M'] else 0
                Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class Model(nn.Module):
    '''empty'''

    def __init__(self, input_dim=128, in_channels=1, out_channels=4, latent_dim=256):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(
            out_channels, out_channels * 2, 3, padding='same')
        self.flatten_dim = (input_dim // 4) ** 2 * out_channels * 2
        self.fc1 = nn.Linear(self.flatten_dim, self.flatten_dim//4)
        self.fc2 = nn.Linear(self.flatten_dim//4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)
        self.max = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''empty'''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max(x)
        x = x.view(-1, self.flatten_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def tss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    score = (tp / (tp + fn + EPS)) - (fp / (fp + tn + EPS))
    return score


def hss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + EPS
    score = numerator / denominator
    return score


def predict(modalities, y, loss_fn, model):
    '''empty'''
    with torch.no_grad():
        out = model(modalities)
        loss = loss_fn(out, y).item()
        return out, loss


def fit(train_loader, val_loader, loss_fn, model, optimizer, lr_scheduler, n_epochs):
    '''empty'''
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    min_val_loss = torch.tensor(float('inf'))
    lr_strikes = 0
    best_model_state_dict = None
    best_model_epoch = 0
    stats = {}
    for epoch in range(1, n_epochs+1):
        train_loss = 0
        train_acc = 0
        model.train()
        for batch in train_loader:
            modalities, y = batch
            out = model(modalities)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            out[out >= .5] = 1
            out[out < .5] = 0
            train_acc += torch.sum(out == y).item() / len(y)
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))
        lr_strikes += 1

        val_loss = 0
        val_acc = 0
        model.eval()
        for batch in val_loader:
            modalities, y = batch
            out, loss = predict(modalities, y, loss_fn, model)
            val_loss += loss
            out[out >= .5] = 1
            out[out < .5] = 0
            val_acc += torch.sum(out == y).item() / len(y)
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))
        print(
            f"epoch {epoch:>3}\ttrain loss: {train_losses[-1]:0.5f}\tval loss: {val_losses[-1]:0.5f}\ttrain acc: {train_accs[-1]:0.5f}\tval acc: {val_accs[-1]:0.5f}")

        if val_losses[-1] < min_val_loss:
            min_val_loss = val_losses[-1]
            lr_strikes = 0
            best_model_state_dict = deepcopy(model.state_dict())
            best_model_epoch = epoch

        if lr_strikes > 5:
            lr_scheduler.step()
            print('reducing lr to', optimizer.param_groups[0]['lr'])
            lr_strikes = 0

    stats['train_losses'] = train_losses
    stats['train_accs'] = train_accs
    stats['val_losses'] = val_losses
    stats['val_accs'] = val_accs

    return model, best_model_state_dict, best_model_epoch, stats


class CustomDataset(Dataset):
    def __init__(self, modalities, labels):
        self.x = modalities
        self.labels = np.expand_dims(labels, axis=1)
        self.x = np.expand_dims(self.x, axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        label = self.labels[index]
        return x, label


class CustomSampler(Sampler):
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
        length = self.negative_length * \
            2 if self.positive_length < self.negative_length else self.positive_length * 2
        return length


def collate(batch):
    xs, ys = [], []
    for sample in batch:
        pil, y = sample
        xs.append(pil)
        ys.append(y)
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    return torch.tensor(xs, dtype=torch.float32).to(DEVICE), torch.tensor(ys, dtype=torch.float32).to(DEVICE)


def main():
    if len(sys.argv) != 5:
        raise ValueError(
            'usage: python main.py {partitions root} {training partitions} {validation partitions} {testing partitions}\nexample: python main.py /data 135 2 4')
    root, training_partitions, validation_partitions, testing_partitions = sys.argv[1:]
    training_paths = [os.path.join(
        root, f'partition{path}') for path in training_partitions]
    val_paths = [os.path.join(
        root, f'partition{path}') for path in validation_partitions]
    test_paths = [os.path.join(
        root, f'partition{path}') for path in testing_partitions]

    print('using', DEVICE)
    x_train, y_train = get_raw_data(training_paths)
    x_val, y_val = get_raw_data(val_paths)
    x_test, y_test = get_raw_data(test_paths)

    n_epochs = 2_000
    batch_size = 128
    lr = 1e-7

    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    test_dataset = CustomDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=CustomSampler(y_train), collate_fn=collate)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=CustomSampler(y_val), collate_fn=collate)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate)

    model = Model().to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)

    model, best_model_state_dict, best_model_epoch, stats = fit(
        train_loader, val_loader, criterion, model, optimizer, lr_scheduler, n_epochs)

    torch.save(best_model_state_dict, os.path.join(
        '/home/mmml_pil/out/prediction/unimodal/convexhull', f'best_{best_model_epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(
        '/home/mmml_pil/out/prediction/unimodal/convexhull', f'last_{n_epochs}.pth'))

    model.load_state_dict(best_model_state_dict)

    y_true, y_pred = [], []
    model.eval()
    for batch in val_loader:
        modalities, y = batch
        out, _ = predict(modalities, y, criterion, model)
        y_pred.extend(list(out.cpu().detach().numpy().squeeze()))
        y_true.extend(list(y.cpu().detach().numpy().squeeze()))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print('optimal threshold:', optimal_threshold)

    test_loss = 0
    y_true, y_pred = [], []
    model.eval()
    for batch in test_loader:
        modalities, y = batch
        out, loss = predict(modalities, y, criterion, model)
        test_loss += loss
        out[out >= optimal_threshold] = 1
        out[out < optimal_threshold] = 0
        y_pred.extend(list(out.cpu().detach().numpy().squeeze()))
        y_true.extend(list(y.cpu().detach().numpy().squeeze()))
    test_losses = [None for _ in range(n_epochs)]
    test_losses[-1] = test_loss / len(test_loader)
    test_hsss = [None for _ in range(n_epochs)]
    test_hsss[-1] = hss(y_pred, y_true)
    test_tsss = [None for _ in range(n_epochs)]
    test_tsss[-1] = tss(y_pred, y_true)
    stats['test_losses'] = test_losses
    stats['test_hsss'] = test_hsss
    stats['test_tsss'] = test_tsss

    df = pd.DataFrame(stats)
    df.to_csv('/home/mmml_pil/out/prediction/unimodal/convexhull/stats.csv')


if __name__ == '__main__':
    main()
