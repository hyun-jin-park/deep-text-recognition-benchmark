import os
import lmdb
from glob import glob
from create_lmdb_dataset import checkImageIsValid, writeCache, load_classes_dictionary


def get_gt_from_file_name(file_name, classes):
    label = ''
    try:
        name = '.'.join(file_name.split('.')[:-1])
        name = name.split('_L_')[1]
        for ch in name:
            if classes.get(ch) is None:
                print('unknown class: ' + ch)
                label += '<UNK>'
            else:
                label += ch
    except IndexError:
        print(f'{file_name} has index error')
        raise IndexError
    return label


def is_valid_label(label, classes):
    for ch in label:
        if classes.get(ch) is None:
            return False
    return True


def copy_lmdb(src_lmdb_env, target_lmdb_env, classes):
    data_cache = {}
    with target_lmdb_env.begin(write=False) as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None:
            print('target lmdb is empty')
            target_count = 0
        else:
            target_count = int(num_samples)

    with src_lmdb_env.begin(write=False) as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None:
            print('src lmdb is empty')
            exit()
        src_count = int(num_samples)

    with src_lmdb_env.begin(write=False) as src_txn:
        for idx in range(0, src_count):
            image_key = 'image-%09d'.encode() % idx
            label_key = 'label-%09d'.encode() % idx
            label = src_txn.get(label_key)
            if label is None:
                print(f'{label} has no data')
                continue

            label = label.decode('utf-8')
            label = label.replace(',', '.')
            if not is_valid_label(label, classes):
                print(f'{label} has invalid label ')
                continue

            image_bin = src_txn.get(image_key)
            image_key = 'image-%09d'.encode() % target_count
            label_key = 'label-%09d'.encode() % target_count
            data_cache[label_key] = label.encode()
            data_cache[image_key] = image_bin
            target_count += 1

            if target_count % 1000 == 1:
                writeCache(target_lmdb_env, data_cache)

        data_cache['num-samples'.encode()] = str(target_count - 1).encode()
        writeCache(target_lmdb_env, data_cache)
        print(f'target dataset size: {target_count}')


def merge_dataset(src1_lmdb_path, src2_lmdb_path, target_lmdb_path):
    os.makedirs(target_lmdb_path, exist_ok=True)
    src1_lmdb_env = lmdb.open(src1_lmdb_path, map_size=1099511627776)
    src2_lmdb_env = lmdb.open(src2_lmdb_path, map_size=1099511627776)
    target_lmdb_env = lmdb.open(target_lmdb_path, map_size=1099511627776)
    classes = load_classes_dictionary('kr_labels.txt')
    copy_lmdb(src1_lmdb_env, target_lmdb_env, classes)
    copy_lmdb(src2_lmdb_env, target_lmdb_env, classes)

def renew_dataset(src_lmdb_path, target_lmdb_path):
    os.makedirs(target_lmdb_path, exist_ok=True)
    src1_lmdb_env = lmdb.open(src_lmdb_path, map_size=1099511627776)
    target_lmdb_env = lmdb.open(target_lmdb_path, map_size=1099511627776)
    classes = load_classes_dictionary('kr_labels.txt')
    copy_lmdb(src1_lmdb_env, target_lmdb_env, classes)

if __name__ == '__main__':
    # merge_dataset('data/real_data_revised/500_real_data', 'data/evaluation/march_real_data_good')
    # merge_dataset('data/real_data_revised/2500_real_data', 'data/evaluation/march_real_data_good', 'data/real_data_lmdb_0412')
    # merge_dataset('data/real_data_revised/500_real_data', 'data/real_data_lmdb_0412', 'data/real_data_lmdb_0412_v2')
    #  merge_dataset('data/real_data_revised/1200_real_data', 'data/real_data_lmdb_0412_v2', 'data/real_data_lmdb_0412_v3')
    # merge_dataset('data/evaluation/real_data_lmdb', 'data/real_data_lmdb_0412_v3', 'data/real_data_lmdb_0412_final')
    # merge_dataset('added_month_day_lmdb', 'data/train/unity', 'unity_added_month_day_lmdb')
    renew_dataset('data/evaluation/real_month_day_march_lmdb', 'data/real_month_day_march_lmdb_renew')