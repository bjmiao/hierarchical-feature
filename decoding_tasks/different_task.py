from builtins import FileNotFoundError
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import logging
import time
import tensorflow as tf
import multiprocessing
from sklearn.model_selection import train_test_split
from mydecoder import DenseNNClassification, LinearClassification
import os
import SharedArray as sa
import string
import random
import argparse


base_path = "/home/bjmiao/Documents/hierarchical-feature/cachedata/production/"
# STIMS = ['nn', 'tex', 'noise']
# REGIONS = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']

def decoding_wrapper(gpu_id, X_train, y_train, X_test, y_test, message,
                     units = 128, dropout = 0., num_epochs = 30, logfile = "training.log", verbose = 0, save_model_path = "./", tag = 'default'):
    logging.info(f"Units = {units} num_epochs = {num_epochs}, dropout = {dropout} tag = {tag}")

    # if use_cpu:
    #     decoder = DenseNNClassification(units=units, num_epochs=num_epochs, dropout=dropout, verbose=verbose, logfile=logfile)
    #     decoder.fit(X_train, y_train, X_test, y_test)
    #     y_predict = decoder.predict(X_test)
    #     correct_count, accuracy = (y_predict == y_test).sum(), (y_predict == y_test).sum() / (len(y_predict))
    # else:
    with tf.device(f'/device:GPU:{gpu_id}'):
        decoder = DenseNNClassification(units=units, num_epochs=num_epochs, dropout=dropout, verbose=verbose, logfile=logfile)
        decoder.fit(X_train, y_train, X_test, y_test)
        if verbose == 1:
            decoder.model.save(os.path.join(save_model_path, tag))
        np.save(os.path.join(save_model_path, "X_train.npy"), X_train)
        np.save(os.path.join(save_model_path, "X_test.npy"), X_test)
        np.save(os.path.join(save_model_path, "y_train.npy"), y_train)
        np.save(os.path.join(save_model_path, "y_test.npy"), y_test)

        y_predict = decoder.predict(X_test)
        correct_count, accuracy = (y_predict == y_test).sum(), (y_predict == y_test).sum() / (len(y_predict))
        
    message = message.replace('[Correct]', str(correct_count)).replace('[Accuracy]', str(accuracy))
    logging.info(message)
    logging.info("y_test = " + str(list(y_test)))
    logging.info("y_predict = " + str(list(y_predict)))

def linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, save_model_path, tag):
    decoder = LinearClassification()
    decoder.fit(X_train, y_train)


    # generating random strings
    rndstr = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=4))
    

    decoder.dump_coef(os.path.join(save_model_path, f"{tag}_{rndstr}.npy"))
    y_predict = decoder.predict(X_test)
    correct_count, accuracy = (y_predict == y_test).sum(), (y_predict == y_test).sum() / (len(y_predict))

    message = message.replace('[Correct]', str(correct_count)).replace('[Accuracy]', str(accuracy))
    logging.info(message)
    logging.info("y_test = " + str(list(y_test)))
    logging.info("y_predict = " + str(list(y_predict)))

def load_npz(filename):
    """
    load npz files with sparse matrix and dimension
    output dense matrix with the correct dim
    """
    npzfile = np.load(filename, allow_pickle=True) 
    sparse_matrix = npzfile['arr_0'][0]
    ndim=npzfile['arr_0'][1]

    new_matrix_2d = np.array(sparse_matrix.todense())
    new_matrix = new_matrix_2d.reshape(ndim)
    return new_matrix

def load_session(mouse_id, block_id, base_path = "/home/bjmiao/Documents/hierarchical-feature/cachedata/production/"):
    img_meta = pd.read_csv(f'/data/decipher_variability/variability_phase4/visual_stimuli/production/images_{block_id}.csv')
    mouse_path = os.path.join(base_path, f"{mouse_id}_production1")
    if not os.path.exists(mouse_path):
        raise RuntimeError(f"produciton1 for {mouse_id} not exists")
    npy_filename = os.path.join(mouse_path, f"{block_id}_tiff_cortex_nwb2_rep.npz")
    print(npy_filename)
    cortical_units_filename = os.path.join(mouse_path, "cortical_units.csv")
    df_cortical_units = pd.read_csv(cortical_units_filename)
    matrix = load_npz(npy_filename)
    print(mouse_id, matrix.shape)
    return matrix


def make_datasets(X_raw, y_raw, nsample_units = -1):
    '''
        X_raw: (neuron * stim * rep * time)
        y_raw: (neuron, )
    '''
    nunits, nstim, nreps, ntimes = X_raw.shape
    X_stim_sample = None
    if nsample_units != -1:  # subsample of the whole population
        random_index = np.random.choice(nunits, nsample_units, replace=False)
        X_stim_sample = X_raw[random_index, :] # get subsample from a large matrix
    else:
        X_stim_sample = X_raw # do not need sampling use whole population
    X_stim_sample = X_stim_sample.reshape(nsample_units, nstim * nreps, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
    X_stim_sample = X_stim_sample.reshape(X_stim_sample.shape[0], -1) # (nsamples, nfeatures)
    print(X_stim_sample.shape, y_raw.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_stim_sample, y_raw, test_size = 0.2)
    return X_train, X_test, y_train, y_test

# def task1_session(units, zscore_normalize = False):
#     '''
#         Do task1 decoding using units in each session.
#         This proved to be not effective.
#     '''
#     mouseIDs = ['594534', '594584', '594585', '597503', '597504', '597505', '597506', '597507',
#                 '598431', '599894', '602518', '607187', '607188', '610740', '610746', '612803']
#     pool = multiprocessing.Pool(processes = 1)
#     stim_map_mapping = {'blank': 0, 'nn': 1, 'tex': 2, 'noise': 3}
#     for mouse_id in mouseIDs:
#         try:
#             matrix = load_session(mouse_id, 'block1_1')
#         except:
#             continue
#         print(matrix.shape)
#         cortical_units_filename = os.path.join(base_path, f"{mouse_id}_production1","cortical_units.csv")
#         df_cortical_units = pd.read_csv(cortical_units_filename)

#         for region, df_one_region in df_cortical_units.groupby("ccf"):
#             X_stim_all = matrix[df_one_region.index]
#             nunits, nstim, nreps, ntimes = X_stim_all.shape
#             X_stim_all = X_stim_all.reshape(nunits, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
#             X_stim_all = X_stim_all.reshape(X_stim_all.shape[0], -1) # (nsamples, nfeatures)
#             y_stim = [ stim_map_mapping[stim] for stim in img_meta.stim_type ]
            
#             y_stim = np.repeat(np.array(y_stim), nreps)

#             X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim, zscore_normalize=zscore_normalize)
            
#             print("X_train.shape=", X_train.shape)
#             print("X_test.shape=", X_test.shape)
#             print("y_train.shape=", y_train.shape)
#             print("y_test.shape=", y_test.shape)
            
#             message = (f"Classify stim_type in {region}, task={task_name}, mouse_id={mouse_id}, units={units}, zcore={zscore_normalize}, Correct=[Correct], Accuracy=[Accuracy]")
#             p = multiprocessing.Process(target = decoding_wrapper, args = (gpu_id, X_train, y_train, X_test, y_test, message, units))
#             p.start()
#             p.join()

def generate_task1_dataset(all_blocks_region_matrix, region, BLOCKS, start_timebin, stop_timebin):
    stim_map_mapping = {'nn': 0, 'tex': 1, 'noise': 2}
    X_stim_all = []
    y_stim_all = []
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        img_meta_related_stim = img_meta[img_meta.stim_type.isin(stim_map_mapping.keys())]
        stim_index = img_meta_related_stim.index
        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index, :, start_timebin:stop_timebin] # use only 50-150ms feature
        y_stim = [ stim_map_mapping[stim] for stim in img_meta_related_stim.stim_type ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    return X_stim_all, y_stim_all

def generate_task2_dataset(all_blocks_region_matrix, region, stim, BLOCKS, start_timebin, stop_timebin):
    subcategory_all = []
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        print(img_meta.stim_type.unique())
        img_meta = img_meta[img_meta['stim_type'] == stim]
        print(img_meta)
        img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
        # print(img_meta['sc'])
        # img_meta['sc'].unique()
        subcategory = img_meta['sc'].unique()
        # img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        subcategory_all += list(subcategory)
    subcategory_all = set(subcategory_all)
    subcategory_all = sorted(set(subcategory_all))
    subcategory_matching = {subcategory: index for (index, subcategory) in enumerate(subcategory_all)}
    print(subcategory_matching)
    print("img_names_matching finished", len(subcategory_matching))

    X_stim_all = []
    y_stim_all = []
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
        stim_index_subset = img_meta[img_meta['stim_type'] == stim].index

        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index_subset, :, start_timebin:stop_timebin] # use only 50-150ms feature
        y_stim = [ subcategory_matching[img] for img in img_meta.iloc[stim_index_subset]['sc'] ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
        print(X_stim.shape)
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    return X_stim_all, y_stim_all

def generate_task2_dataset_allstim(all_blocks_region_matrix, region, stims, BLOCKS, start_timebin, stop_timebin):
    subcategory_all = []
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        print(img_meta.stim_type.unique())
        img_meta = img_meta[img_meta['stim_type'].isin(stims)]
        print(img_meta)
        img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
        # print(img_meta['sc'])
        # img_meta['sc'].unique()
        subcategory = img_meta['sc'].unique()
        # img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        subcategory_all += list(subcategory)
    subcategory_all = set(subcategory_all)
    subcategory_all = sorted(set(subcategory_all))
    subcategory_matching = {subcategory: index for (index, subcategory) in enumerate(subcategory_all)}
    print(subcategory_matching)
    print("img_names_matching finished", len(subcategory_matching))

    X_stim_all = []
    y_stim_all = []
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
        stim_index_subset = img_meta[img_meta['stim_type'].isin(stims)].index

        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index_subset, :, start_timebin:stop_timebin] # use only 50-150ms feature
        y_stim = [ subcategory_matching[img] for img in img_meta.iloc[stim_index_subset]['sc'] ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
        print(X_stim.shape)
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    return X_stim_all, y_stim_all

def generate_task3_dataset(all_blocks_region_matrix, region, stim, BLOCKS, start_timebin, stop_timebin):
    img_names_all = []
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        img_names_all += list(img_names)
    img_names_all = list(enumerate(img_names_all))
    img_names_matching = {v: k for (k, v) in img_names_all}
    print("img_names_matching finished", len(img_names_matching))
    X_stim_all = []
    y_stim_all = []
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        stim_index_subset = img_meta[img_meta['stim_type'] == stim].index
        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index_subset, :, start_timebin:stop_timebin] # use only 50-150ms feature
        y_stim = [ img_names_matching[img] for img in img_meta.iloc[stim_index_subset]['unique_img'] ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
        print(X_stim.shape)
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    return X_stim_all, y_stim_all

def generate_task3_dataset_allstim(all_blocks_region_matrix, region, stims, BLOCKS, start_timebin, stop_timebin):
    img_names_all = []
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        img_names = img_meta[img_meta['stim_type'].isin(stims)]['unique_img'].unique()
        img_names_all += list(img_names)
    img_names_all = list(enumerate(img_names_all))
    img_names_matching = {v: k for (k, v) in img_names_all}
    print("img_names_matching finished", len(img_names_matching))
    X_stim_all = []
    y_stim_all = []
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        stim_index_subset = img_meta[img_meta['stim_type'].isin(stims)].index
        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index_subset, :, start_timebin:stop_timebin] # use only 50-150ms feature
        y_stim = [ img_names_matching[img] for img in img_meta.iloc[stim_index_subset]['unique_img'] ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
        print(X_stim.shape)
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    return X_stim_all, y_stim_all

def task1_linear(args, all_blocks_region_matrix):
    task = "task1_linear"
    for region in args.regions:
        start_timebin, stop_timebin = args.start_timebin, args.stop_timebin
        X_stim_all, y_stim_all = generate_task1_dataset(all_blocks_region_matrix, region, args.blocks, start_timebin, stop_timebin)

        # start bootstraping
        nsampling_times = args.nsamples
        for nsample_units in args.nsample_units_list:
            for _ in range(nsampling_times):
                X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)
                print("X_train.shape=", X_train.shape)
                print("X_test.shape=", X_test.shape)
                print("y_train.shape=", y_train.shape)
                print("y_test.shape=", y_test.shape)
                
                message = f"Linear decoder for task={task} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, args.save_model_path, f"{region}_task1_{nsample_units}")

def task2_linear_within_stim(args, all_blocks_region_matrix):
    task = "task2_linear_within_stim"
    for region in args.regions:
        for stim in args.stims:
            start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
            X_stim_all, y_stim_all = generate_task2_dataset(all_blocks_region_matrix, region, stim, args.blocks, start_timebin, stop_timebin)

            for nsample_units in args.nsample_units_list:
                for _ in range(args.nsamples):
                    X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                    print("X_train.shape=", X_train.shape)
                    print("X_test.shape=", X_test.shape)
                    print("y_train.shape=", y_train.shape)
                    print("y_test.shape=", y_test.shape)
                    
                    message = f"Linear decoder for task={task} stim={stim} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                    linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, args.save_model_path, f"{region}_{stim}_{task}_{nsample_units}")

def task2_linear_all_stim(args, all_blocks_region_matrix):
    task = "task2_linear_within_stim"
    for region in args.regions:
        start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
        X_stim_all, y_stim_all = generate_task2_dataset_allstim(all_blocks_region_matrix, region, args.stims, args.blocks, start_timebin, stop_timebin)

        for nsample_units in args.nsample_units_list:
            for _ in range(args.nsamples):
                X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                print("X_train.shape=", X_train.shape)
                print("X_test.shape=", X_test.shape)
                print("y_train.shape=", y_train.shape)
                print("y_test.shape=", y_test.shape)
                
                message = f"Linear decoder for task={task} stim={args.stims} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, args.save_model_path, f"{region}_{args.stims}_{task}_{nsample_units}")


def task3_linear_within_stim(args, all_blocks_region_matrix):
    task = "task3_linear_within_stim"
    for region in args.regions:
        for stim in args.stims:
            start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
            X_stim_all, y_stim_all = generate_task3_dataset(all_blocks_region_matrix, region, stim, args.blocks, start_timebin, stop_timebin)

            for nsample_units in args.nsample_units_list:
                for _ in range(args.nsamples):
                    X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                    print("X_train.shape=", X_train.shape)
                    print("X_test.shape=", X_test.shape)
                    print("y_train.shape=", y_train.shape)
                    print("y_test.shape=", y_test.shape)
                    
                    message = f"Linear decoder for task={task} stim={stim} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                    linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, args.save_model_path, f"{region}_{stim}_{task}_{nsample_units}")

def task3_linear_all_stim(args, all_blocks_region_matrix):
    task = "task3_linear_all_stim"
    for region in args.regions:
        start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
        X_stim_all, y_stim_all = generate_task3_dataset_allstim(all_blocks_region_matrix, region, args.stims, args.blocks, start_timebin, stop_timebin)

        for nsample_units in args.nsample_units_list:
            for _ in range(args.nsamples):
                X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                print("X_train.shape=", X_train.shape)
                print("X_test.shape=", X_test.shape)
                print("y_train.shape=", y_train.shape)
                print("y_test.shape=", y_test.shape)
                
                message = f"Linear decoder for task={task} stim={args.stims} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                linear_decoder_wrapper(X_train, y_train, X_test, y_test, message, args.save_model_path, f"{region}_{args.stims}_{task}_{nsample_units}")

def task2_dnn_within_stim(args, all_blocks_region_matrix):
    task = "task2_dnn_within_stim"
    for region in args.regions:
        for stim in args.stims:
            start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
            X_stim_all, y_stim_all = generate_task2_dataset(all_blocks_region_matrix, region, stim, args.blocks, start_timebin, stop_timebin)

            for nsample_units in args.nsample_units_list:
                for _ in range(args.nsamples):
                    X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                    print("X_train.shape=", X_train.shape)
                    print("X_test.shape=", X_test.shape)
                    print("y_train.shape=", y_train.shape)
                    print("y_test.shape=", y_test.shape)
                    
                    message = f"DNN decoder for task={task} stim={stim} region={region} nsample_units={nsample_units} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                    p = multiprocessing.Process(target = decoding_wrapper, args = (args.gpu_id, X_train, y_train, X_test, y_test, message))
                    p.start()
                    p.join()

def task2_dnn_within_stim_tuning(args, all_blocks_region_matrix):
    task = "task2_dnn_within_stim_tuning"
    for region in args.regions:
        for stim in args.stims:
            start_timebin, stop_timebin = args.start_timebin, args.stop_timebin            
            X_stim_all, y_stim_all = generate_task2_dataset(all_blocks_region_matrix, region, stim, args.blocks, start_timebin, stop_timebin)

            for nsample_units in args.nsample_units_list:
                for _ in range(args.nsamples):
                    X_train, X_test, y_train, y_test = make_datasets(X_stim_all, y_stim_all, nsample_units)

                    X_train = X_train.astype("float64")
                    X_test = X_test.astype("float64")
                    print("X_train.shape=", X_train.shape, X_train.dtype)
                    print("X_test.shape=", X_test.shape, X_test.dtype)
                    print("y_train.shape=", y_train.shape, y_train.dtype)
                    print("y_test.shape=", y_test.shape, y_test.dtype)
                    # import pdb
                    # pdb.set_trace()

                    # # for units in [8, 16, 32, 64, 128, 256, 400, 512]:
                    # for units in [512, 128, 32]:
                    for units in [512]:
                        for dropout in [0]:
                        # for dropout in [0, 0.2]:
                            epoch = 3
                            training_logfile = f"log_zscore_tuning/training_{stim}_{region}_{nsample_units}_model_{units}_{dropout}.log"
                            message = f"DNN tuning decoder for task={task} stim={stim} region={region} nsample_units={nsample_units} units={units} dropout={dropout} epoch={epoch} start_time_bin={start_timebin} stop_time_bin={stop_timebin} Correct=[Correct] Accuracy=[Accuracy]"
                            p = multiprocessing.Process(target = decoding_wrapper, args = (args.gpu_id, X_train, y_train, X_test, y_test, message, units, dropout, epoch, training_logfile))
                            p.start()
                            p.join()


def task1_bootstraping(regions,
            bootstraping_units = 500, sampling_times = 10, can_repeat = False,
            zscore_normalize = False):
    stim_map_mapping = {'blank': 0, 'nn': 1, 'tex': 2, 'noise': 3}
    for region in regions:
        X_stim_all = []
        y_stim_all = []
        for block_id in BLOCKS:
            all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
            matrix = all_region_matrix[region]
            X_stim = matrix
            y_stim = [ stim_map_mapping[stim] for stim in img_meta.stim_type ]
            X_stim_all.append(X_stim)
            y_stim_all += y_stim

        X_stim_all = np.concatenate(X_stim_all, axis=1)
        nunits, nstim, nreps, ntimes = X_stim_all.shape
        y_stim_all = np.repeat(np.array(y_stim_all), nreps)

        # start bootstraping
        for _ in range(sampling_times):
            if can_repeat == True:
                random_index = np.random.choice(nunits, bootstraping_units)
            else:
                random_index = np.random.choice(nunits, bootstraping_units, replace=False)
            X_stim_sample = X_stim_all[random_index, :] # get subsample from a large matrix
            X_stim_sample = X_stim_sample.reshape(bootstraping_units, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
            X_stim_sample = X_stim_sample.reshape(X_stim_sample.shape[0], -1) # (nsamples, nfeatures)
            print(X_stim_sample.shape, y_stim_all.shape)
            X_train, X_test, y_train, y_test = make_datasets(X_stim_sample, y_stim_all, zscore_normalize)
            print("X_train.shape=", X_train.shape)
            print("X_test.shape=", X_test.shape)
            print("y_train.shape=", y_train.shape)
            print("y_test.shape=", y_test.shape)
            
            message = f"Classify stim_type in {region}, task={task_name}, " \
                    + f"bootstraping_units={bootstraping_units}, zcore={zscore_normalize}, " \
                    + "Correct=[Correct], Accuracy=[Accuracy]"
            p = multiprocessing.Process(target = decoding_wrapper, args = (gpu_id, X_train, y_train, X_test, y_test, message))
            p.start()
            p.join()

def task1(zscore_normalize = False):
    '''
        classify stim_type

        Sample: task1(gpu_id = 1)
    '''
    logging.info("Task1:  classify stim_type (largest category). It also combines all the blocks")
    print(f"gpu_id = {gpu_id}")
    pool = multiprocessing.Pool(processes = 1)

    # img_names_all = []
    # task_name = f"classify_sample_for_{stim}"
    # for block_id in BLOCKS:
    #     _, img_meta = all_blocks_region_matrix[block_id]
    #     img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
    #     img_names_all += list(img_names)
    # img_names_all = list(enumerate(img_names_all))
    # img_names_matching = {v: k for (k, v) in img_names_all}
    # print("img_names_matching finished", len(img_names_matching))
    stim_map_mapping = {'blank': 0, 'nn': 1, 'tex': 2, 'noise': 3}
    # do simulation for each region
    for region in REGIONS:
        X_stim_all = []
        y_stim_all = []
        for block_id in BLOCKS:
            all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
            matrix = all_region_matrix[region]
            X_stim = matrix
            y_stim = [ stim_map_mapping[stim] for stim in img_meta.stim_type ]
            X_stim_all.append(X_stim)
            y_stim_all += y_stim
        X_stim_all = np.concatenate(X_stim_all, axis=1)
        nunits, nstim, nreps, ntimes = X_stim_all.shape
        X_stim_all = X_stim_all.reshape(nunits, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
        X_stim_all = X_stim_all.reshape(X_stim_all.shape[0], -1) # (nsamples, nfeatures)
        y_stim_all = np.repeat(np.array(y_stim_all), nreps)
        print(X_stim_all.shape, y_stim_all.shape)
        X_train, X_test, y_train, y_test = train_test_split(X_stim_all, y_stim_all)
        if zscore_normalize == True:
            logging.info("Using zscore normalization")
            train_mean, train_stdev = X_train.mean(axis = 0), X_train.std(axis = 0)
            train_mean, train_stdev = np.expand_dims(train_mean, 0), np.expand_dims(train_stdev, 0)
            print(train_mean.shape, train_stdev.shape)
            X_train = (X_train - train_mean)/train_stdev
            X_train = np.nan_to_num(X_train)

            X_test = (X_test - train_mean) / train_stdev
            X_test = np.nan_to_num(X_test)
            print(X_train.mean(0), X_test.mean(0), X_train.std(axis = 0), X_test.std(axis = 0))

        print("X_train.shape=", X_train.shape)
        print("X_test.shape=", X_test.shape)
        print("y_train.shape=", y_train.shape)
        print("y_test.shape=", y_test.shape)
        
        message = (f"Classify stim_type in {region}, task={task_name}, zscore={zscore_normalize}, Correct=[Correct], Accuracy=[Accuracy]")
        p = multiprocessing.Process(target = decoding_wrapper, args = (gpu_id, X_train, y_train, X_test, y_test, message))
        p.start()
        p.join()

def task2_within_stim(stim, bootstraping_units = 500, sampling_times = 10, can_repeat = False):
    '''
        This task is to classify sub-category in each stim. It also combines all the blocks
        Sample: task2_within_stim(stim = 'tex')
    '''
    print(f"gpu_id = {gpu_id}, stim = {stim}")
    task_name = f"task2_classify_sample_within_{stim}"
    subcategory_all = []
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        img_meta = img_meta[img_meta['stim_type'] == stim]
        img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
        # print(img_meta['sc'])
        # img_meta['sc'].unique()
        subcategory = img_meta['sc'].unique()
        # img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        subcategory_all += list(subcategory)
    subcategory_all = set(subcategory_all)
    subcategory_all = sorted(set(subcategory_all))
    subcategory_matching = {subcategory: index for (index, subcategory) in enumerate(subcategory_all)}
    print(subcategory_matching)
    print("img_names_matching finished", len(subcategory_matching))

    # do simulation for each region
    for region in REGIONS:
        X_stim_all = []
        y_stim_all = []
        for block_id in BLOCKS:
            all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
            img_meta['sc'] = img_meta['stim_type'] + "_" + img_meta['category']
            stim_index_subset = img_meta[img_meta['stim_type'] == stim].index
            matrix = all_region_matrix[region]
            X_stim = matrix[:, stim_index_subset, :, :]
            y_stim = [ subcategory_matching[img] for img in img_meta.iloc[stim_index_subset]['sc'] ]
            X_stim_all.append(X_stim)
            y_stim_all += y_stim
            print(X_stim.shape)
        X_stim_all = np.concatenate(X_stim_all, axis=1)
        nunits, nstim, nreps, ntimes = X_stim_all.shape
        y_stim_all = np.repeat(np.array(y_stim_all), nreps)

        # start bootstraping
        for _ in range(sampling_times):
            if can_repeat == True:
                random_index = np.random.choice(nunits, bootstraping_units)
            else:
                random_index = np.random.choice(nunits, bootstraping_units, replace=False)
            X_stim_sample = X_stim_all[random_index, :] # get subsample from a large matrix
            X_stim_sample = X_stim_sample.reshape(bootstraping_units, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
            X_stim_sample = X_stim_sample.reshape(X_stim_sample.shape[0], -1) # (nsamples, nfeatures)
            print(X_stim_sample.shape, y_stim_all.shape)
            X_train, X_test, y_train, y_test = make_datasets(X_stim_sample, y_stim_all)
            print("X_train.shape=", X_train.shape)
            print("X_test.shape=", X_test.shape)
            print("y_train.shape=", y_train.shape)
            print("y_test.shape=", y_test.shape)
        
            message = (f"Category in stim={stim}, region={region}, task={task_name},  Correct=[Correct], Accuracy=[Accuracy]")
            p = multiprocessing.Process(target = decoding_wrapper, args = (gpu_id, X_train, y_train, X_test, y_test, message))
            p.start()
            p.join()

def task3_tuning(region, stim):
    img_names_all = []
    task_name = f"tuning_for_region_{stim}_stim_{stim}"
    logging.info(task_name)
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        img_names_all += list(img_names)
    img_names_all = list(enumerate(img_names_all))
    img_names_matching = {v: k for (k, v) in img_names_all}
    print("img_names_matching finished", len(img_names_matching))

    X_stim_all = []
    y_stim_all = []
    # do simulation for each region
    for block_id in BLOCKS:
        all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
        stim_index_subset = img_meta[img_meta['stim_type'] == stim].index
        matrix = all_region_matrix[region]
        X_stim = matrix[:, stim_index_subset, :, :]
        y_stim = [ img_names_matching[img] for img in img_meta.iloc[stim_index_subset]['unique_img'] ]
        X_stim_all.append(X_stim)
        y_stim_all += y_stim
        print(X_stim.shape)
    X_stim_all = np.concatenate(X_stim_all, axis=1)
    nunits, nstim, nreps, ntimes = X_stim_all.shape
    X_stim_all = X_stim_all.reshape(nunits, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
    X_stim_all = X_stim_all.reshape(X_stim_all.shape[0], -1) # (nsamples, nfeatures)
    y_stim_all = np.repeat(np.array(y_stim_all), nreps)
    print(X_stim_all.shape, y_stim_all.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_stim_all, y_stim_all)
    print("X_train.shape=", X_train.shape)
    print("X_test.shape=", X_test.shape)
    print("y_train.shape=", y_train.shape)
    print("y_test.shape=", y_test.shape)
    
    for units in [8, 16, 32, 64, 128, 256, 400, 512]:
        for dropout in [0, 0.2, 0.4, 0.5, 0.6, 0.8]:
            epoch = 50
            message = f"Category in stim={stim}, region={region}, task={task_name}, units={units}, dropout={dropout}, epoch={epoch}, Correct=[Correct], Accuracy=[Accuracy]"
            logfile = f"log/training_{region}_{stim}_{units}_{dropout}.log"
            p = multiprocessing.Process(
                target = decoding_wrapper,
                args = (gpu_id, X_train, y_train, X_test, y_test, message, units, dropout, epoch, logfile)
            )
            p.start()
            p.join()

def task3_within_stim(stim, bootstraping_units = 500, sampling_times = 10, can_repeat = False):
    '''
        This task is to classify samples in different stim. It combines all the blocks")

        Sample: task3_within_stim(stim = 'tex')
    '''
    logging.info("This task is to classify samples among different stim. It combines all the blocks")
    print(f"gpu_id = {gpu_id}, stim = {stim}")

    img_names_all = []
    task_name = f"classify_sample_for_{stim}"
    for block_id in BLOCKS:
        _, img_meta = all_blocks_region_matrix[block_id]
        img_names = img_meta[img_meta['stim_type'] == stim]['unique_img'].unique()
        img_names_all += list(img_names)
    img_names_all = list(enumerate(img_names_all))
    img_names_matching = {v: k for (k, v) in img_names_all}
    print("img_names_matching finished", len(img_names_matching))

    # do simulation for each region
    for region in REGIONS:
        X_stim_all = []
        y_stim_all = []
        for block_id in BLOCKS:
            all_region_matrix, img_meta = all_blocks_region_matrix[block_id]
            stim_index_subset = img_meta[img_meta['stim_type'] == stim].index
            matrix = all_region_matrix[region]
            X_stim = matrix[:, stim_index_subset, :, :]
            y_stim = [ img_names_matching[img] for img in img_meta.iloc[stim_index_subset]['unique_img'] ]
            X_stim_all.append(X_stim)
            y_stim_all += y_stim
            print(X_stim.shape)
        X_stim_all = np.concatenate(X_stim_all, axis=1)
        nunits, nstim, nreps, ntimes = X_stim_all.shape
        y_stim_all = np.repeat(np.array(y_stim_all), nreps)

        # start bootstraping
        for _ in range(sampling_times):
            if can_repeat == True:
                random_index = np.random.choice(nunits, bootstraping_units)
            else:
                random_index = np.random.choice(nunits, bootstraping_units, replace=False)
            X_stim_sample = X_stim_all[random_index, :] # get subsample from a large matrix
            X_stim_sample = X_stim_sample.reshape(bootstraping_units, -1, ntimes).transpose([1, 0, 2]) # (nsampels, nunits, ntimes)
            X_stim_sample = X_stim_sample.reshape(X_stim_sample.shape[0], -1) # (nsamples, nfeatures)
            print(X_stim_sample.shape, y_stim_all.shape)
            X_train, X_test, y_train, y_test = make_datasets(X_stim_sample, y_stim_all)
            print("X_train.shape=", X_train.shape)
            print("X_test.shape=", X_test.shape)
            print("y_train.shape=", y_train.shape)
            print("y_test.shape=", y_test.shape)
        
            message = (f"Category in stim={stim}, region={region}, task={task_name}, bootstraping_units={bootstraping_units}, Correct=[Correct], Accuracy=[Accuracy]")
            p = multiprocessing.Process(target = decoding_wrapper, args = (gpu_id, X_train, y_train, X_test, y_test, message))
            p.start()
            p.join()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("task", help="which task you are running")
    parser.add_argument("--regions", type=str, nargs='+', default=['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam'], help = "Run which region")
    parser.add_argument("--blocks", type=str, nargs='+', default=['block1_1', 'block1_2', 'block2_2'], help = "Aggregate data on which blocks")
    parser.add_argument("--stims", type=str, nargs='+', default=['nn', 'tex', 'noise'], help = "Decode within which stimulus. For task2 task 3")

    parser.add_argument('--use_zscore', action="store_true", help="Should use zscore data")
    parser.add_argument("--use_cpu", action = "store_true", help = "Do we use cpu tensorflow when training NN")
    parser.add_argument("--gpu_id", type=int, default=0, help="Run on which gpu (can be only one number)")

    parser.add_argument("--start_timebin", default=50, type=int)
    parser.add_argument("--stop_timebin", default=150, type=int)
    parser.add_argument("--nsample_units_list", "-n", type=int, nargs="+", default=[500], help = "a list of sample units")
    parser.add_argument("--nsamples", default=10, type=int, help="Repeat experiments how many times for each nsample_units")

    parser.add_argument("--logfile", default="neural_decoding", type=str, help="Logfile name prefix")
    parser.add_argument("--save_model_path", default="saved_model/linear", type=str, help="Linear weight file")
    args = parser.parse_args()
    return args

def get_all_blocks_data(REGIONS, BLOCKS):
    start = time.time()
    all_blocks_region_matrix = {}
    for block_id in BLOCKS:
        img_meta = pd.read_csv(f'/data/decipher_variability/variability_phase4/visual_stimuli/production/images_{block_id}.csv') 
        all_region_matrix = {}
        for region in REGIONS:
            if 1:
            # if args.use_zscore:
            #     try:
            #         all_region_matrix[region] = sa.attach(f"shm://spike_zscore_{block_id}_{region}")
            #     except FileNotFoundError:
            #         print("Not found share memory. Try load raw file")
            #         all_region_matrix[region] = np.load(f"/vault/bjmiao/matrix/production1_zscore_{block_id}_{region}.npy")
            # else:
                try:
                    all_region_matrix[region] = sa.attach(f"shm://spike_rate_{block_id}_{region}")
                except FileNotFoundError:
                    print("Not found share memory. Try load raw file")
                    all_region_matrix[region] = np.load(os.path.join(base_path, f"matrix/production1_{block_id}_{region}.npy"))
        all_blocks_region_matrix[block_id] = (all_region_matrix, img_meta)
    stop = time.time()
    print(f"Load data complete. Used {stop - start} seconds")
    return all_blocks_region_matrix

if __name__ == "__main__":
    args = parse_args()
    print(args)
    try:
        os.makedirs(args.save_model_path)
    except FileExistsError:
        pass

    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    logging.basicConfig(filename=f"/home/bjmiao/Documents/hierarchical-feature/decoding_tasks/{args.logfile}_{today}.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    all_blocks_region_matrix = get_all_blocks_data(args.regions, args.blocks)

    task_func = globals()[args.task]
    task_func(args, all_blocks_region_matrix)

    # # np.random.seed(19971008)
    # # task1(gpu_id = gpu_id, stim = stim)
    # task_name = sys.argv[1]
    # print(task_name)
    # # if task_name == "task1":
    # #     gpu_id = int(sys.argv[2])
    # #     # task1(gpu_id, zscore_normalize = False)
    # #     task1(zscore_normalize = True)
    # # if task_name == "task2":
    # #     gpu_id = int(sys.argv[2])
    # #     task2()
    # if task_name == "task1_session":
    #     units = int(sys.argv[2])
    #     zscore_normalize = int(sys.argv[3])
    #     if zscore_normalize:
    #         zscore_normalize = True
    #     else:
    #         zscore_normalize = False
    #     task1_session(units = units, zscore_normalize = zscore_normalize)
    # if task_name == "task1_linear":
    #     task1_linear()
    # if task_name == "task2_linear_within_stim":
    #     stim = sys.argv[2]
    #     task2_linear_within_stim(stim)
    # if task_name == "task3_linear_within_stim":
    #     stim = sys.argv[2]
    #     task3_linear_within_stim(stim)
    # if task_name == "task1_bootstraping":
    #     task1_bootstraping(
    #         regions = REGIONS,
    #         bootstraping_units = 1000, sampling_times = 10, can_repeat = False,
    #         zscore_normalize = False
    #     )
    # if task_name == "task2_within_stim":
    #     stim = sys.argv[2]
    #     task2_within_stim(stim)
    # if task_name == "task3_within_stim":
    #     stim = sys.argv[2]
    #     task3_within_stim(stim = stim)