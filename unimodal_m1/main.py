import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from aeon.classification.dictionary_based import MUSE
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier


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
    X = np.asarray(X, dtype=np.float128)
    mean = np.mean(X, axis=(0, 1))
    std = np.std(X, axis=(0, 1))
    X = (X - mean) / std
    X = np.asarray(X, dtype=np.float32)
    return X, mean, std


def acc(yhat, y):
    '''empty'''
    return np.mean(yhat == y)


def tss(yhat, y):
    '''empty'''
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    score = (tp / (tp + fn)) - (fp / (fp + tn))
    return score


def hss(yhat, y):
    '''empty'''
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    score = numerator / denominator
    return score


def main():
    '''empty'''
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
    x_train, y_train = get_raw_data(training_paths)
    x_val, y_val = get_raw_data(val_paths)
    x_test, y_test = get_raw_data(test_paths)

    x_train = interpolate_properties(x_train)
    x_val = interpolate_properties(x_val)
    x_test = interpolate_properties(x_test)

    x_train = np.transpose(x_train, [0, 2, 1])
    x_val = np.transpose(x_val, [0, 2, 1])
    x_test = np.transpose(x_test, [0, 2, 1])

    nf_ix = np.where(y_train == 0)[0]
    f_ix = np.where(y_train == 1)[0]

    undersampled_nf_train_ix = np.random.choice(nf_ix, len(f_ix), False)
    x_train_balanced = x_train[np.append(f_ix, undersampled_nf_train_ix)]
    y_train_balanced = y_train[np.append(f_ix, undersampled_nf_train_ix)]

    model_1 = TimeSeriesForestClassifier()
    model_2 = RDSTClassifier()
    model_3 = MUSE()
    model_4 = KNeighborsTimeSeriesClassifier()

    stats = {}

    print('training 1')
    yhat_train_1 = model_1.fit_predict(x_train_balanced, y_train_balanced)

    print('training 2')
    yhat_train_2 = model_2.fit_predict(x_train_balanced, y_train_balanced)

    print('training 3')
    yhat_train_3 = model_3.fit_predict(x_train_balanced, y_train_balanced)

    print('training 4')
    yhat_train_4 = model_4.fit_predict(x_train_balanced, y_train_balanced)

    print('validating 1')
    yhat_val_1 = model_1.predict(x_val)

    print('validating 2')
    yhat_val_2 = model_2.predict(x_val)

    print('validating 3')
    yhat_val_3 = model_3.predict(x_val)

    print('validating 4')
    yhat_val_4 = model_4.predict(x_val)

    print('testing 1')
    yhat_test_1 = model_1.predict(x_test)

    print('testing 2')
    yhat_test_2 = model_2.predict(x_test)

    print('testing 3')
    yhat_test_3 = model_3.predict(x_test)

    print('testing 4')
    yhat_test_4 = model_4.predict(x_test)

    stats['train_1_acc'] = acc(yhat_train_1, y_train_balanced)
    stats['train_2_acc'] = acc(yhat_train_2, y_train_balanced)
    stats['train_3_acc'] = acc(yhat_train_3, y_train_balanced)
    stats['train_4_acc'] = acc(yhat_train_4, y_train_balanced)

    stats['val_1_tss'] = tss(yhat_val_1, y_val)
    stats['val_2_tss'] = tss(yhat_val_2, y_val)
    stats['val_3_tss'] = tss(yhat_val_3, y_val)
    stats['val_4_tss'] = tss(yhat_val_4, y_val)

    stats['val_1_hss'] = hss(yhat_val_1, y_val)
    stats['val_2_hss'] = hss(yhat_val_2, y_val)
    stats['val_3_hss'] = hss(yhat_val_3, y_val)
    stats['val_4_hss'] = hss(yhat_val_4, y_val)

    stats['test_1_tss'] = tss(yhat_test_1, y_test)
    stats['test_2_tss'] = tss(yhat_test_2, y_test)
    stats['test_3_tss'] = tss(yhat_test_3, y_test)
    stats['test_4_tss'] = tss(yhat_test_4, y_test)

    stats['test_1_hss'] = hss(yhat_test_1, y_test)
    stats['test_2_hss'] = hss(yhat_test_2, y_test)
    stats['test_3_hss'] = hss(yhat_test_3, y_test)
    stats['test_4_hss'] = hss(yhat_test_4, y_test)
    
    pprint(stats)

    df = pd.DataFrame(stats, index=[0])
    df.to_csv('unimodal_properties_results.csv', index=False)


if __name__ == '__main__':
    main()
