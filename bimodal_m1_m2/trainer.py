import os
import sys
import util
import torch
import model
import training
import numpy as np
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    '''empty'''
    if len(sys.argv) != 8:
        raise ValueError(
            'usage: python trainer.py {partitions root} {training partitions} {validation partitions} {# epochs} {output path} {properties encoder path} {pil encoder path}\nexample: python trainer.py /data 135 2 1000 ./output ./properties.pt ./pil.pt')
    root, training_partitions, validation_partitions, n_epochs_str, output_path, properties_enc_path, pil_enc_path = sys.argv[
        1:]
    training_paths = [os.path.join(
        root, f'partition{path}') for path in training_partitions]
    val_paths = [os.path.join(
        root, f'partition{path}') for path in validation_partitions]
    OUTPUT_PATH = os.path.join(output_path, datetime.strftime(
        datetime.now(), '%Y-%m-%d %H:%M:%S'))

    N_EPOCHS = int(n_epochs_str)

    os.mkdir(OUTPUT_PATH)
    os.mkdir(os.path.join(OUTPUT_PATH, 'models'))
    os.mkdir(os.path.join(OUTPUT_PATH, 'losses'))

    X_train_properties, X_train_pil, _, y_train = util.get_raw_data(
        training_paths)
    X_val_properties, X_val_pil, _, y_val = util.get_raw_data(val_paths)

    X_train_properties = util.interpolate_properties(X_train_properties)
    X_val_properties = util.interpolate_properties(X_val_properties)

    X_train_properties, mean_train, std_train = util.z_normalization(
        X_train_properties)
    X_val_properties = (X_val_properties - mean_train) / std_train
    X_val_properties = np.asarray(X_val_properties, np.float32)

    train_dataset = util.CustomDataset(
        [X_train_properties, X_train_pil], y_train)
    val_dataset = util.CustomDataset(
        [X_val_properties, X_val_pil], y_val)

    train_loader = DataLoader(train_dataset, 256, sampler=util.PILSampler(
        y_train), collate_fn=util.collate)
    val_loader = DataLoader(val_dataset, 256, sampler=util.PILSampler(
        y_val), collate_fn=util.collate)

    properties_enc = model.PropertiesEncoder(18, 256, 128, 64, device)
    properties_enc_state = torch.load(properties_enc_path, map_location=device)
    properties_enc.load_state_dict(properties_enc_state)

    pil_enc = model.RasterEncoder()
    pil_state = torch.load(pil_enc_path, map_location=device)
    pil_enc.load_state_dict(pil_state)

    inference_model = model.InferenceModel(
        properties_enc, pil_enc, 128, device).to(device)

    criterion = torch.nn.BCELoss().to(device)
    optimizer = optim.Adam(inference_model.parameters(), lr=1e-7)
    optimizer.param_groups[0]['params'] = [
        param.to(device) for param in optimizer.param_groups[0]['params']]
    lr_sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    train_losses, val_losses, _ = training.fit(train_loader, val_loader, inference_model,
                                               criterion, optimizer, lr_sched, 5, N_EPOCHS, os.path.join(OUTPUT_PATH, 'models'), device)

    np.save(os.path.join(OUTPUT_PATH, 'losses', 'train.npy'),
            np.array(train_losses, dtype=np.float32))
    np.save(os.path.join(OUTPUT_PATH, 'losses', 'val.npy'),
            np.array(val_losses, dtype=np.float32))


if __name__ == '__main__':
    main()
