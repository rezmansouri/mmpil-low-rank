import os
import sys
import util
import torch
import model
import train
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    '''empty'''
    if len(sys.argv) != 6:
        raise ValueError(
            'usage: python main.py {partitions root} {training partitions} {validation partitions} {# epochs} {output path}\nexample: python main.py /data 13 5 1000 ./output')
    root, training_partitions, validation_partitions, n_epochs_str, output_path = sys.argv[1:]
    training_paths = [os.path.join(
        root, f'partition{path}') for path in training_partitions]
    val_paths = [os.path.join(
        root, f'partition{path}') for path in validation_partitions]
    OUTPUT_PATH = os.path.join(output_path, datetime.strftime(
        datetime.now(), '%Y-%m-%d %H:%M:%S'))
    os.mkdir(OUTPUT_PATH)
    os.mkdir(os.path.join(OUTPUT_PATH, 'models'))
    os.mkdir(os.path.join(OUTPUT_PATH, 'losses'))

    X_train, _ = util.get_raw_data(training_paths)
    X_train = util.interpolate_properties(X_train)

    X_val, _ = util.get_raw_data(val_paths)
    X_val = util.interpolate_properties(X_val)

    X_train, mean_train, std_train = util.z_normalization(X_train)
    X_val = (X_val - mean_train) / std_train
    X_val = np.asarray(X_val, np.float32)

    train_dataset = util.CustomDataset(X_train)
    train_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=True, collate_fn=util.batch_second)
    val_dataset = util.CustomDataset(X_val)
    val_loader = DataLoader(val_dataset, batch_size=512,
                            shuffle=True, collate_fn=util.batch_second)

    N_FEATURES = 18
    N_EPOCHS = int(n_epochs_str)
    H1 = 1024
    H2 = H1 // 2
    H3 = H2 // 2

    encoder = model.Encoder(N_FEATURES, H1, H2, H3, device).to(device)
    decoder_lstm_cell_1 = nn.LSTMCell(N_FEATURES, H1).to(device)
    decoder_lstm_cell_2 = nn.LSTMCell(H1, H2).to(device)
    decoder_lstm_cell_3 = nn.LSTMCell(H2, H3).to(device)
    linear = nn.Linear(H3, N_FEATURES).to(device)

    criterion = nn.SmoothL1Loss()
    params = list(encoder.parameters()) + \
        list(decoder_lstm_cell_1.parameters()) + \
        list(decoder_lstm_cell_2.parameters()) + \
        list(decoder_lstm_cell_3.parameters()) + \
        list(linear.parameters())

    optimizer = optim.Adam(params, lr=1e-3)
    optimizer.param_groups[0]['params'] = [
        param.to(device) for param in optimizer.param_groups[0]['params']]
    lr_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)

    train_losses, val_losses, _ = train.fit(train_loader, val_loader, encoder, decoder_lstm_cell_1, decoder_lstm_cell_2, decoder_lstm_cell_3, linear,
                                            criterion, optimizer, lr_sched, 5, N_EPOCHS, os.path.join(OUTPUT_PATH, 'models'), device)

    np.save(os.path.join(OUTPUT_PATH, 'losses', 'train.npy'),
            np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(OUTPUT_PATH, 'losses', 'val.npy'),
            np.array(val_losses, dtype=np.float32))


if __name__ == '__main__':
    main()
