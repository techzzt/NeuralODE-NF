import numpy as np

from skimage.measure import block_reduce # for downsample
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.nn import functional as F

def downsample_data(list_arr, downsample, reduce_method=np.median, all_should_be_anomaly=True):
    # list_arr contains a list of array with the first dimension as downsample target
    # assume if there are only 0, 1 in an array, then it is the labels array
    new_list_arr = []
    for arr in list_arr:
        if len(arr.shape) == 1 and (arr == 0).sum() + (arr == 1).sum() == np.prod(arr.shape):
            reduce_func = np.min if all_should_be_anomaly else np.max
            block_sizes = (downsample, )
        else:
            reduce_func = reduce_method
            block_sizes = (downsample, 1)
        new_len = len(arr) // downsample * downsample
        arr = block_reduce(arr[:new_len], block_sizes, reduce_func) 
        new_list_arr.append(arr)

    return new_list_arr

def preprocess(dat, config, dims, entities, entity_id, quiet=False):
    assert 'x_trn' in dat.keys() and 'x_tst' in dat.keys() and 'lab_tst' in dat.keys()
    x_trn, x_tst, lab_tst = dat['x_trn'], dat['x_tst'], dat['lab_tst']
  
    assert type(x_trn) is np.ndarray and type(x_tst) is np.ndarray and type(lab_tst) is np.ndarray
    assert x_trn.shape[-1] == dims and x_tst.shape[-1] == dims
    assert len(x_trn.shape) == 2 and len(x_tst.shape) == 2 and len(lab_tst.shape) == 1

    if not quiet:
        print('Original shapes x_trn, x_tst, lab_tst=', x_trn.shape, x_tst.shape, lab_tst.shape)

    # delete channels whose std = 0 for both trn and val
    if 'entities' not in vars(config):
        keep_chns = (x_trn.std(axis=0) + x_tst.std(axis=0)) > 0
        x_trn = x_trn[:, keep_chns]
        x_tst = x_tst[:, keep_chns]
        print('Deleted channels with std = 0 x_trn, x_tst=', x_trn.shape, x_tst.shape)

    # apply normalization
    normalizer = MinMaxScaler(feature_range=(-1, 1)).fit(x_trn)
    x_trn = normalizer.transform(x_trn)
    x_tst = normalizer.transform(x_tst)
    if not quiet:
        print(f'Applied minmax normalization')

    # clamp test data
    assert x_trn.max() <= 1 + 1e-10 and x_trn.min() >= -1 - 1e-10
    if config.clamp_max is not None:
        x_tst[x_tst > config.clamp_max] = config.clamp_max
    if config.clamp_min is not None:
        x_tst[x_tst < config.clamp_min] = config.clamp_min

    # downsample
    if config.downsample is not None and config.downsample > 1:
        x_trn, x_tst, lab_tst = downsample_data([x_trn, x_tst, lab_tst], config.downsample)
        if not quiet:
            print('Downsampled x_trn, x_tst, lab_tst=', x_trn.shape, x_tst.shape, lab_tst.shape)
    if not quiet:
        all_a = (lab_tst == 1).sum()
        print(f'AR = {all_a/len(lab_tst):.4f} ({all_a} / {len(lab_tst)})')

    # round the length of the dataset such that dl + stride * K = length, where K is an integer
    x_trn = x_trn[:len(x_trn) - (len(x_trn)-config.dl) % config.stride]
    tst_stride = config.dl if config.tst_stride == 'no_rep' else config.tst_stride
    x_tst = x_tst[:len(x_tst) - (len(x_tst)-config.dl) % tst_stride]
    lab_tst = lab_tst[:len(lab_tst) - (len(lab_tst)-config.dl) % tst_stride]
    if not quiet:
        print('Rounded x_trn, x_tst, lab_tst=', x_trn.shape, x_tst.shape, lab_tst.shape)

    assert (len(x_trn) - config.dl) % config.stride == 0
    assert len(x_tst) % tst_stride == 0

    if entities > 1 and config.train_method == 'train_together':
        entity = np.eye(entities)[entity_id]
        x_trn = np.concatenate((x_trn, np.tile(entity, (len(x_trn), 1))), axis=-1)
        x_tst = np.concatenate((x_tst, np.tile(entity, (len(x_tst), 1))), axis=-1)
    
    return {'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst}

def window_stride(x_trn_all, x_tst_all, lab_tst_all, entities, dl, stride, tst_stride):
    x_trn, x_tst, lab_tst = [], [], []
    if tst_stride == 'no_rep':
        # No repetition... just follow exactly what is in lab_tst
        assert len(x_tst_all) == len(lab_tst_all)
        idx = 0
        for x_tr in x_trn_all:
            # These are training data
            st = 0
            while st + dl <= len(x_tr):
                x_trn.append(x_tr[st:st+dl])
                st += stride
        for x_te, lab_te in zip(x_tst_all, lab_tst_all):
            # now testing data
            st = 0
            while st + dl <= len(x_te):
                x_tst.append(x_te[st:st+dl])
                lab_tst.append(lab_te[st:st+dl])
                st += tst_stride
    else:
        assert type(x_trn_all) == list and type(x_tst_all) == list and type(lab_tst_all) == list
        # Construct the sliding content
        # need to individually construct every array in the list
        for run in x_trn_all:
            st = 0
            while st + dl <= len(run):
                x_trn.append(run[st:st+dl])
                st += stride
        for run in x_tst_all:
            st = 0
            while st + dl <= len(run):
                x_tst.append(run[st:st+dl])
                st += tst_stride
        for run in lab_tst_all:
            st = 0
            while st + dl <= len(run):
                lab_tst.append(run[st:st+dl])
                st += tst_stride

    # 배열의 형태가 일관되지 않은 경우를 처리
    try:
        # 일단 기존 방식으로 시도
        x_trn, x_tst, lab_tst = np.stack(x_trn, 0), np.stack(x_tst, 0), np.stack(lab_tst, 0)
    except ValueError as e:
        print("Warning: Arrays have inconsistent shapes. Attempting to fix...")
        
        # 1. 각 배열의 형태 확인
        x_trn_shapes = [x.shape for x in x_trn]
        x_tst_shapes = [x.shape for x in x_tst]
        lab_tst_shapes = [x.shape for x in lab_tst]
        
        print(f"x_trn shapes: {set(x_trn_shapes)}")
        print(f"x_tst shapes: {set(x_tst_shapes)}")
        print(f"lab_tst shapes: {set(lab_tst_shapes)}")
        
        # 2. 가장 일반적인 모양 찾기 (가장 많이 나타나는 모양)
        from collections import Counter
        
        # x_trn 처리
        if len(set(x_trn_shapes)) > 1:
            common_shape = Counter(x_trn_shapes).most_common(1)[0][0]
            print(f"Standardizing x_trn to shape {common_shape}")
            
            # 모든 배열을 같은 크기로 조정
            standardized_x_trn = []
            for x in x_trn:
                if x.shape == common_shape:
                    standardized_x_trn.append(x)
            x_trn = standardized_x_trn
        
        # x_tst 처리
        if len(set(x_tst_shapes)) > 1:
            common_shape = Counter(x_tst_shapes).most_common(1)[0][0]
            print(f"Standardizing x_tst to shape {common_shape}")
            
            standardized_x_tst = []
            standardized_lab_tst = []
            
            for i, x in enumerate(x_tst):
                if i < len(lab_tst) and x.shape == common_shape:
                    standardized_x_tst.append(x)
                    standardized_lab_tst.append(lab_tst[i])
            
            x_tst = standardized_x_tst
            lab_tst = standardized_lab_tst
        
        # 다시 스택 시도
        x_trn, x_tst, lab_tst = np.stack(x_trn, 0), np.stack(x_tst, 0), np.stack(lab_tst, 0)
        print("Standardization complete.")
    
    print(f'Window/strided x_trn [=] {x_trn.shape}; x_tst [=] {x_tst.shape}; lab_tst [=] {lab_tst.shape}')

    return {'trn': x_trn_all, 'tst': x_tst_all, 'lab': lab_tst_all, 'x_trn': x_trn, 'x_tst': x_tst, 'lab_tst': lab_tst, 'num_entity': entities}
