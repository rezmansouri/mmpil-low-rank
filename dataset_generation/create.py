import os
import sys
import json
import h5py
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime, timedelta

PROPERTIES_DTYPE = np.dtype([('key', 'S20'), ('value', 'S20')])
EMPTY_PROPERTIES = np.empty(18, dtype=PROPERTIES_DTYPE)
EMPTY_PROPERTIES['key'] = [b'humoment_5', b'humoment_4', b'ratio', b'convexhull_area',
                           b'humoment_7', b'extent_ch', b'convexhull_totalContourArea',
                           b'extent_mobr', b'humoment_3', b'convexhull_totalContourPerimeter',
                           b'humoment_6', b'pil_area', b'pil_count', b'min_aspect_ratio',
                           b'extent_mrbr', b'extent_be', b'humoment_1', b'humoment_2']
EMPTY_PROPERTIES['value'] = ''
EMPTY_INSTANCE = {'harp': None, 'time': None, 'properties': [],
                  'class': None, 'pil': None, 'convexhull': None}


def get_config(path='config.json'):
    with open(path, 'r', encoding='utf-8') as cfg:
        config = json.load(cfg)
    h, m, s = [int(x) for x in config['observation_window'].split(':')]
    config['observation_window'] = timedelta(hours=h, minutes=m, seconds=s)
    h, m, s = [int(x) for x in config['prediction_window'].split(':')]
    config['prediction_window'] = timedelta(hours=h, minutes=m, seconds=s)
    h, m, s = [int(x) for x in config['increment'].split(':')]
    config['increment'] = timedelta(hours=h, minutes=m, seconds=s)
    h, m, s = [int(x) for x in config['time_step'].split(':')]
    config['time_step'] = timedelta(hours=h, minutes=m, seconds=s)
    h, m, s = [int(x) for x in config['lag'].split(':')]
    config['lag'] = timedelta(hours=h, minutes=m, seconds=s)
    return config


def extract(path, ix):
    print(f'Extracting {ix}...')
    os.mkdir(f'temp/in_{ix}')
    with tarfile.open(f'{path}/in_{ix}.tar.gz', 'r') as tar:
        tar.extractall(path=f'temp/in_{ix}')


def get_harpnum_noaa_ars_map(path):
    map_ = dict()
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file.readlines()):
            if i == 0:
                continue
            harpnum, noaa_ars = line.split(' ')
            noaa_ars = noaa_ars.split(',')
            map_[int(harpnum)] = [int(n) for n in noaa_ars]
    return map_


def get_sdo_flares_df(path, time_format):
    df = pd.read_csv(path)
    df['start_time'] = pd.to_datetime(df['start_time'], format=time_format)
    return df


def get_full_seq(h5_names, time_step, time_format):
    result = []
    h5_names = sorted(h5_names)
    trec = datetime.strptime(h5_names[0].split('.')[3][:-4], time_format)
    for i, h5_name in enumerate(h5_names):
        file_time = datetime.strptime(
            h5_names[i].split('.')[3][:-4], time_format)
        if trec + time_step == file_time:
            result.append(h5_name)
            trec = trec + time_step
        else:
            while trec != file_time:
                result.append(None)
                trec = trec + time_step
            result.append(h5_name)
    return result


def is_trustable(lineage_metadata):
    if len(lineage_metadata) == 0:
        return False
    lon_min, lon_max = float(lineage_metadata[7][1].decode(
        'utf-8')), float(lineage_metadata[11][1].decode('utf-8'))
    return -70 < (lon_min + lon_max) / 2 < 70


def get_class(harpnum, time, harpnum_noaa_ars_map, sdo_goes_flares_df, lag, pred_window, ranked_classes):
    lowerbound = time
    upperbound = time + lag + pred_window
    if harpnum not in harpnum_noaa_ars_map:
        return 'FQ'
    noaas = harpnum_noaa_ars_map[harpnum]
    select = sdo_goes_flares_df[(sdo_goes_flares_df['start_time'] >= lowerbound) & (
        sdo_goes_flares_df['start_time'] <= upperbound) & (sdo_goes_flares_df['noaa_active_region'].isin(noaas))]
    if len(select) == 0:
        return 'FQ'
    select_classes = [ranked_classes.index(c[0]) for c in select['goes_class']]
    return ranked_classes[max(select_classes)]


def collect_harp(path, harpnum, observation_window, increment, time_step, max_missing, time_format,
                 harpnum_noaa_ars_map, sdo_goes_flares_df, lag, pred_window, ranked_classes):
    instances = []
    path += f'/{harpnum}'
    file_names = os.listdir(path)
    if len(file_names) == 0:
        return instances
    h5_sequence = get_full_seq(file_names, time_step, time_format)
    sequence_length = observation_window // time_step
    step = increment // time_step
    needle = 0
    candidate_sequence = h5_sequence[needle:needle+sequence_length]
    while len(candidate_sequence) == sequence_length:
        instance = deepcopy(EMPTY_INSTANCE)
        missing = 0
        for i, h5_name in enumerate(candidate_sequence):
            if h5_name is None:
                if len(instance['properties']) == 0:
                    needle += i + 1
                    candidate_sequence = h5_sequence[needle:needle +
                                                     sequence_length]
                    break
                missing += 1
                if missing > max_missing or i == sequence_length-1:
                    needle += i + 1
                    candidate_sequence = h5_sequence[needle:needle +
                                                     sequence_length]
                    break
                instance['properties'].append(EMPTY_PROPERTIES)
                continue
            with h5py.File(f'{path}/{h5_name}', 'r') as h5:
                lineage_metadata = h5.get('lineage_metadata')[:]
                if not is_trustable(lineage_metadata):
                    needle += i + 1
                    candidate_sequence = h5_sequence[needle:needle +
                                                     sequence_length]
                    break
                if 'properties' not in h5:
                    if len(instance['properties']) == 0:
                        needle += i + 1
                        candidate_sequence = h5_sequence[needle:needle +
                                                         sequence_length]
                        break
                    missing += 1
                    if missing > max_missing or i == sequence_length-1:
                        needle += i + 1
                        candidate_sequence = h5_sequence[needle:needle +
                                                         sequence_length]
                        break
                    instance['properties'].append(EMPTY_PROPERTIES)
                    continue
                instance['properties'].append(h5.get('properties')[:])
                if i == sequence_length-1:
                    file_time = h5_name.split('.')[3][:-4]
                    instance['pil'] = h5.get('pil')[:]
                    instance['convexhull'] = h5.get('convexhull')[:]
                    trec = datetime.strptime(file_time, time_format)
                    instance['class'] = get_class(
                        harpnum, trec, harpnum_noaa_ars_map, sdo_goes_flares_df, lag, pred_window, ranked_classes)
                    instance['harp'] = harpnum
                    instance['time'] = file_time
                    instance['properties'] = np.array(
                        instance['properties'], dtype=PROPERTIES_DTYPE)
                    instances.append(instance)
                    needle += step
                    candidate_sequence = h5_sequence[needle:needle +
                                                     sequence_length]
    return instances


def collect(path, observation_window, increment, time_step, max_missing, time_format,
            harpnum_noaa_ars_map, sdo_goes_flares_df, lag, pred_window, ranked_classes):
    print(f'Collecting... {path}')
    instances = []
    for harpnum in tqdm(sorted(os.listdir(path)), leave=False):
        ins = collect_harp(path, int(harpnum), observation_window, increment, time_step, max_missing, time_format,
                           harpnum_noaa_ars_map, sdo_goes_flares_df, lag, pred_window, ranked_classes)
        instances.extend(ins)
    return instances


def write(instances, path):
    print(f'Writing to {path}...')
    os.mkdir(f'{path}/out')
    for instance in tqdm(instances):
        file_path = f"{path}/out/{instance['harp']}.e{instance['time']}.h5"
        with h5py.File(file_path, 'w') as file:
            for key, value in instance.items():
                if key == 'harp' or key == 'time':
                    continue
                file.create_dataset(key, data=value)


def main():
    if len(sys.argv) < 3:
        print(
            'usage: python main.py {input directory path} {output directory path}')
        print('input directory path: directory of harp number directories containing h5 instances')
        print('output directory path: path of the resulting collected instances. a directory named out will be created there.')
        raise ValueError
    in_path = sys.argv[1]
    if not os.path.exists(in_path):
        raise ValueError('input directory path does not exist')
    out_path = sys.argv[2]
    if not os.path.exists(out_path):
        raise ValueError('output directory path does not exist')
    config = get_config()
    harpnum_noaa_ars_map = get_harpnum_noaa_ars_map(
        config['harpnum_noaa_ars_map_path'])
    sdo_flares_df = get_sdo_flares_df(
        config['flare_list_path'], config['flare_list_time_format'])
    instances = collect(in_path, config['observation_window'], config['increment'], config['time_step'], config['max_allowed_missing_time_step'],
                        config['data_time_format'], harpnum_noaa_ars_map, sdo_flares_df, config['lag'], config['prediction_window'], config['ranked_classes'])
    write(instances, out_path)


if __name__ == '__main__':
    main()
