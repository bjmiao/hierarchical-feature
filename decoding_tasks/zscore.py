import SharedArray as sa
import numpy as np
import os
import time
base_path = "/home/bjmiao/Documents/hierarchical-feature/cachedata/production/"


def zscore_transform(matrix):
    ''' we do zscore to matrix on the acis of neuron.
        each neuron's response overall the session will be zscore'ed
    '''
    origin_shape = matrix.shape
    matrix = matrix.reshape(matrix.shape[0], -1)
    matrix = (matrix - matrix.mean(1).reshape(-1, 1)) / matrix.std(axis = 1).reshape(-1, 1)
    matrix = matrix.astype('float16')
    matrix = matrix.reshape(origin_shape)
    np.nan_to_num(matrix, copy = False)
    return matrix

for region in ['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']:
     for block_id in ['block1_1', 'block1_2', 'block2_2']:
         start = time.time()
         print(region, block_id)
         a = np.load(os.path.join(base_path, f"matrix/production1_{block_id}_{region}.npy"))
         a = zscore_transform(a)
         zscore_result = sa.create(f"shm://spike_zscore_{block_id}_{region}", shape=a.shape, dtype="float16")
         zscore_result[:] = a[:]
         stop = time.time()
         print("complete. Used time:", stop - start)
