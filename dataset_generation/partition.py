import os
import sys
import shutil
from tqdm import tqdm


def get_pil_harp_numbers(path):
    result = set()
    for partition in os.listdir(path):
        for h5 in os.listdir(f'{path}/{partition}'):
            result.add(h5.split('.')[0])
    return list(result)


def copy(objects_paths, out_path):
    for object_path in objects_paths:
        shutil.copy2(object_path, out_path)


def main():
    if len(sys.argv) != 4:
        raise ValueError(
            'usage: python main.py {pil path} {swan-sf path} {out path}')
    pil_path, swan_sf_path, out_path = sys.argv[1:]
    swan_sf_partition_paths = [
        f'{swan_sf_path}/{l}' for l in os.listdir(swan_sf_path)]
    if len(swan_sf_partition_paths) != 5:
        raise ValueError('wrong swan_sf_path')
    swan_sf_partition_paths = sorted(swan_sf_partition_paths)

    pil_paths = []
    for partition in os.listdir(pil_path):
        pil_paths = pil_paths + [
            f'{pil_path}/{partition}/{l}' for l in os.listdir(f'{pil_path}/{partition}')]
    for i, swan_sf_partition_path in enumerate(tqdm(swan_sf_partition_paths)):
        swan_sf_harps = [l.split('.')[0]
                         for l in os.listdir(swan_sf_partition_path)]
        
        os.mkdir(f'{out_path}/partition{i+1}')
        for swan_sf_harp in swan_sf_harps:
            pil_paths_to_copy = [l for l in pil_paths if f'/{swan_sf_harp}.' in l]
            copy(pil_paths_to_copy, f'{out_path}/partition{i+1}')

if __name__ == '__main__':
    main()
