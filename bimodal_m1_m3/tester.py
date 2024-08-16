import os
import sys
import util
import torch
import model
import testing
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    '''empty'''
    if len(sys.argv) != 7:
        raise ValueError(
            'usage: python tester.py {partitions root} {training partitions} {validation partitions} {testing partitions} {output path} {model path}\nexample: python tester.py /data 135 2 4 ./output ./model.pt')
    root, training_partitions, validation_partitions, test_partitions, output_path, model_path = sys.argv[
        1:]
    training_paths = [os.path.join(
        root, f'partition{path}') for path in training_partitions]
    val_paths = [os.path.join(
        root, f'partition{path}') for path in validation_partitions]
    test_paths = [os.path.join(
        root, f'partition{path}') for path in test_partitions]
    OUTPUT_PATH = os.path.join(output_path, 'bimodal-m1-m3' + datetime.strftime(
        datetime.now(), '%Y-%m-%d %H:%M:%S'))

    os.mkdir(OUTPUT_PATH)

    X_train_properties, _, _, _ = util.get_raw_data(
        training_paths)
    X_val_properties, X_val_ch, _, y_val = util.get_raw_data(val_paths)
    X_test_properties, X_test_ch, _, y_test = util.get_raw_data(
        test_paths)

    X_train_properties = util.interpolate_properties(X_train_properties)
    X_val_properties = util.interpolate_properties(X_val_properties)
    X_test_properties = util.interpolate_properties(X_test_properties)

    X_train_properties, mean_train, std_train = util.z_normalization(
        X_train_properties)
    X_val_properties = (X_val_properties - mean_train) / std_train
    X_val_properties = np.asarray(X_val_properties, np.float32)
    X_test_properties = (X_test_properties - mean_train) / std_train
    X_test_properties = np.asarray(X_test_properties, np.float32)

    val_dataset = util.CustomDataset(
        [X_val_properties, X_val_ch], y_val)
    test_dataset = util.CustomDataset(
        [X_test_properties, X_test_ch], y_test)

    val_loader = DataLoader(val_dataset, 256, sampler=util.PILSampler(
        y_val), collate_fn=util.collate)
    test_loader = DataLoader(test_dataset, 256, collate_fn=util.collate)

    inference_model = torch.load(model_path, map_location=device).to(device)

    testing.test(val_loader, test_loader, inference_model,
                 os.path.join(OUTPUT_PATH, 'results.csv'))


if __name__ == '__main__':
    main()
