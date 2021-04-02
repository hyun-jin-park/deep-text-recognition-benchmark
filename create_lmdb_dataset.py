""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def get_gt_from_file_name(file_name, classes): 
    name = '.'.join(file_name.split('.')[:-1])
    label = ''
    conversion = {'~~Star~~':'*',
                '~~BackSlash~~':'\\',
                '~~Slash~~':'/' ,
                '~~Colon~~':':',
                '~~SemiColon~~':';',
                '__LeftBrace__':'<',
                # '_':'<',
                '~~RightBrace~~':'>',
                '~~underscore~~':'_',
                '~~score~~':'-' ,
                }

    for key, value in conversion.items():
        name = name.replace(key, value)

    # name = name.replace('__LeftBrace__','<')
    # name = name.replace('-', '<')
    name = name.split('_')[0]    
    for ch in name:
        if classes.get(ch) is None:
            print('unknown class: ' + ch)
            label += '<UNK>'
        else:
            label += ch
    return label


def load_classes_dictionary(file_path, include_space=False):
    classes = {} 
    with open (file_path, 'r') as f:
        lines = f.readlines()
        for line in lines: 
            id, label = line.strip().split()
            classes[label] = id
    if include_space :
        classes[' '] = str(int(id) + 1)
    return classes


def createDataset(inputPath, outputPath, checkValid=True, use_space=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
        use_space : if true add space classes
    """
    os.makedirs(outputPath, exist_ok=True)
    env_train = lmdb.open(os.path.join(outputPath, 'train'), map_size=1099511627776)
    env_val = lmdb.open(os.path.join(outputPath, 'val'), map_size=1099511627776)
    classes = load_classes_dictionary(os.path.join(inputPath, 'kr_labels.txt'), use_space)
    cache_train = {}
    cache_val = {}

    with env_train.begin(write=False) as txn :
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None: 
            cnt_train = 1
        else:
            cnt_train = int(num_samples)

    with env_val.begin(write=False) as txn:
        num_samples = txn.get('num-samples'.encode())
        if num_samples is None: 
            cnt_val = 1 
        else:
            cnt_val = int(num_samples)    

    import glob
    import random 

    for image_path in glob.iglob(os.path.join(inputPath, "**"), recursive=True):
        if os.path.isfile(image_path): 
            _ , file_name = os.path.split(image_path)
            label = get_gt_from_file_name(file_name, classes) 
        else:
            continue

        #Current implementation has no consideration for UNK character 
        #This time I decieded to remove samples which have <UNK> character 
        if label.find('<UNK>') >= 0 : 
            continue 

        if len(label) > 30 :
            print(f'skip {label} {len(label)}')
            continue

        with open(image_path, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % image_path)
                    continue
            except:
                print('error occured', image_path)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(image_path))
                continue

        if random.randrange(0, 10) > 0: 
            imageKey = 'image-%09d'.encode() % cnt_train
            labelKey = 'label-%09d'.encode() % cnt_train
            cache_train[imageKey] = imageBin
            cache_train[labelKey] = label.encode()
            if cnt_train % 1000 == 0:
                writeCache(env_train, cache_train)
                cache_train = {}
                print('Written %d' % (cnt_train))
            cnt_train += 1
        else :
            imageKey = 'image-%09d'.encode() % cnt_val
            labelKey = 'label-%09d'.encode() % cnt_val
            cache_val[imageKey] = imageBin
            cache_val[labelKey] = label.encode()
            if cnt_val % 1000 == 0:
                writeCache(env_val, cache_val)
                cache_val = {}
                print('Written %d' % (cnt_val))
            cnt_val += 1

    cache_train['num-samples'.encode()] = str(cnt_train-1).encode()
    writeCache(env_train, cache_train)
    cache_val['num-samples'.encode()] = str(cnt_val-1).encode()
    writeCache(env_val, cache_val)
    print('Created dataset with train: %d val: %dsamples' % (cnt_train-1, cnt_val-1))


if __name__ == '__main__':
    fire.Fire(createDataset)
