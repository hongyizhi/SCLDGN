# -*- coding: utf-8 -*-
# !/usr/bin/env python3

import random

import numpy as np
import mne
import pandas as pd
from scipy.io import loadmat, savemat
import os
import pickle
import csv
from shutil import copyfile
import sys
import resampy
import shutil
import urllib.request as request
from contextlib import closing
from scipy.linalg import sqrtm, inv

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))  # To load all the relevant files
from dataset.eegDataset import eegDataset
from utils import transforms


def parseBci42aFile(dataPath, labelPath, epochWindow=[0, 4], chans=list(range(22))):
    if dataPath[-6:-4] == '4T':
        eventCode = [4]
    else:
        eventCode = [6]
    fs = 250
    offset = 2

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
    eeg = raw_gdf.get_data()

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]
    # Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
    x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

    # Multiply the data with 1e6
    x = x * 1e6

    # Load the labels
    y = loadmat(labelPath)["classlabel"].squeeze()
    # change the labels from [1-4] to [0-3]
    y = y - 1

    x = x.transpose((2, 0, 1))
    x = alignOperation(x, operation='svd')
    # x, y = online_operation(x, y, start=0, n=2)
    x = x.transpose((1, 2, 0))

    data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': fs}
    return data


def parseBci42bFile(dataPath, labelPath, epochWindow=[0, 4], chans=list(range(3))):
    # start of the trial at t=0
    if int(dataPath[-6:-5]) == 1 or int(dataPath[-6:-5]) == 2:
        eventCode = [2]
    else:
        eventCode = [3]
    fs = 250
    offset = 3

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
    eeg = raw_gdf.get_data()

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
    x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

    # Multiply the data with 1e6
    x = x * 1e6

    # Load the labels
    y = loadmat(labelPath)["classlabel"].squeeze()
    # change the labels from [1-4] to [0-3]
    y = y - 1

    data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': fs}
    return data


def parseHGDFile(dataPath, epochWindow=[-0.5, 3.5], chans=list(range(44))):
    rename_dict = {'EEG FC5': 'FC5', 'EEG FC1': 'FC1', 'EEG FC2': 'FC2', 'EEG FC6': 'FC6', 'EEG C3': 'C3',
                   'EEG C4': 'C4', 'EEG CP5': 'CP5', 'EEG CP1': 'CP1', 'EEG CP2': 'CP2', 'EEG CP6': 'CP6',
                   'EEG FC3': 'FC3', 'EEG FCz': 'FCz', 'EEG FC4': 'FC4', 'EEG C5': 'C5', 'EEG C1': 'C1',
                   'EEG C2': 'C2', 'EEG C6': 'C6', 'EEG CP3': 'CP3', 'EEG CPz': 'CPz', 'EEG CP4': 'CP4',
                   'EEG FFC5h': 'FFC5h', 'EEG FFC3h': 'FFC3h', 'EEG FFC4h': 'FFC4h', 'EEG FFC6h': 'FFC6h',
                   'EEG FCC5h': 'FCC5h', 'EEG FCC3h': 'FCC3h', 'EEG FCC4h': 'FCC4h', 'EEG FCC6h': 'FCC6h',
                   'EEG CCP5h': 'CCP5h', 'EEG CCP3h': 'CCP3h', 'EEG CCP4h': 'CCP4h', 'EEG CCP6h': 'CCP6h',
                   'EEG CPP5h': 'CPP5h', 'EEG CPP3h': 'CPP3h', 'EEG CPP4h': 'CPP4h', 'EEG CPP6h': 'CPP6h',
                   'EEG FFC1h': 'FFC1h', 'EEG FFC2h': 'FFC2h', 'EEG FCC1h': 'FCC1h', 'EEG FCC2h': 'FCC2h',
                   'EEG CCP1h': 'CCP1h', 'EEG CCP2h': 'CCP2h', 'EEG CPP1h': 'CPP1h', 'EEG CPP2h': 'CPP2h'}

    C_sensors = [
        'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
        'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
        'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
        'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
        'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
        'CCP2h', 'CPP1h', 'CPP2h']

    # load the gdf file using MNE
    raw_edf = mne.io.read_raw_edf(dataPath, stim_channel="auto")
    raw_edf.load_data()
    raw_edf.rename_channels(rename_dict)
    raw_edf.pick_channels(ch_names=C_sensors, ordered=True)
    raw_edf.set_eeg_reference(ref_channels='average')
    raw_edf.resample(sfreq=250)
    eeg = raw_edf.get_data()

    # 与文章作者一样的处理方式,整段做指数平移标准化
    eeg = eeg * 1e6
    eeg = np.clip(eeg, -800, 800)
    data = exponential_moving_standardize(eeg, 1e-3, 1000)

    eventCode = [0, 1, 2, 3]
    mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}
    gdf_events = mne.events_from_annotations(raw_edf, mapping)[0][:, [0, 2]].tolist()
    events = [event for event in gdf_events if event[1] in eventCode]

    # 标签
    y = np.array([i[1] for i in events])

    # 切分
    fs = 250
    offset = -0.5
    epochWindow = [0, 4]
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + int(offset * fs)

    x = np.stack([data[:, epochInterval + event[0]] for event in events], axis=2)

    x = x.transpose((2, 0, 1))
    x = alignOperation(x, operation='svd')
    # x, y = online_operation(x, y, start=0, n=1)
    x = x.transpose((1, 2, 0))

    data = {'x': x, 'y': y, 'c': C_sensors, 's': fs}
    return data


def parseBci42aDataset(datasetPath, savePath,
                       epochWindow=[0, 4], chans=list(range(22)), verbos=False):
    subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
    test_subjects = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            if not os.path.exists(os.path.join(datasetPath, sub + '.mat')):
                raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                                 os.path.join(datasetPath, sub + '.mat') +
                                 ' Please download and copy the extracted dataset at the above path ' +
                                 ' More details about how to download this data can be found in the Instructions.txt file')

            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))
            data = parseBci42aFile(os.path.join(datasetPath, sub + '.gdf'),
                                   os.path.join(datasetPath, sub + '.mat'),
                                   epochWindow=epochWindow, chans=chans)
            savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)


def parseBci42bDataset(datasetPath, savePath,
                       epochWindow=[0, 4], chans=list(range(3)), verbos=False):
    # "start event = 2 left 3 right 4"
    s1t = ['B0101T', 'B0102T', 'B0103T']
    s1e = ['B0104E', 'B0105E']
    s2t = ['B0201T', 'B0202T', 'B0203T']
    s2e = ['B0204E', 'B0205E']
    s3t = ['B0301T', 'B0302T', 'B0303T']
    s3e = ['B0304E', 'B0305E']
    s4t = ['B0401T', 'B0402T', 'B0403T']
    s4e = ['B0404E', 'B0405E']
    s5t = ['B0501T', 'B0502T', 'B0503T']
    s5e = ['B0504E', 'B0505E']
    s6t = ['B0601T', 'B0602T', 'B0603T']
    s6e = ['B0604E', 'B0605E']
    s7t = ['B0701T', 'B0702T', 'B0703T']
    s7e = ['B0704E', 'B0705E']
    s8t = ['B0801T', 'B0802T', 'B0803T']
    s8e = ['B0804E', 'B0805E']
    s9t = ['B0901T', 'B0902T', 'B0903T']
    s9e = ['B0904E', 'B0905E']
    subjects = [s1t, s2t, s3t, s4t, s5t, s6t, s7t, s8t, s9t]
    test_subjects = [s1e, s2e, s3e, s4e, s5e, s6e, s7e, s8e, s9e]
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        # [subjects, test_subject]
        for iSub, sub in enumerate(subs):
            subData = {}
            # [s1t, s2t, s3t, s4t, s5t, s6t, s7t, s8t, s9t],[s1e, s2e, s3e, s4e, s5e, s6e, s7e, s8e, s9e]
            for iS, su in enumerate(sub):
                # ['B0101T']
                if not os.path.exists(os.path.join(datasetPath, su + '.mat')):
                    raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                                     os.path.join(datasetPath, su + '.mat') +
                                     ' Please download and copy the extracted dataset at the above path ' +
                                     ' More details about how to download this data can be found in the Instructions.txt file')

                print('Processing subject No.: ' + subL[iSubs] + su[1:5])
                data = parseBci42bFile(os.path.join(datasetPath, su + '.gdf'),
                                       os.path.join(datasetPath, su + '.mat'),
                                       epochWindow=epochWindow, chans=chans)
                if subData:
                    subData['x'] = np.concatenate((subData['x'], data['x']), axis=2)
                    subData['y'] = np.concatenate((subData['y'], data['y']), axis=0)
                else:
                    subData = data

            name = subL[iSubs] + sub[0][2:3].zfill(3)
            savemat(os.path.join(savePath, name + '.mat'), subData)


def parseHGDDataset(datasetPath, savePath,
                    epochWindow=[-0.5, 3.5], chans=list(range(44)), verbos=False):
    subjects = list(range(0, 14))
    subAll = [subjects, subjects]
    subL = ['s', 'se']
    subName = {'s': 'train', 'se': 'test'}

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            print('Processing subject No.: ' + subL[iSubs] + str(sub + 1).zfill(3))
            print(os.path.join(datasetPath, subName[subL[iSubs]], str(sub + 1) + '.edf'))
            data = parseHGDFile(
                os.path.join(datasetPath, subName[subL[iSubs]], str(sub + 1) + '.edf'),
                epochWindow=epochWindow, chans=chans)
            print(os.path.join(savePath, subL[iSubs] + str(sub + 1).zfill(3) + '.mat'))
            savemat(os.path.join(savePath, subL[iSubs] + str(sub + 1).zfill(3) + '.mat'), data)


def parseGISTDataset(datasetPath, savePath, ):
    subjects = []
    for i in range(1, 53):
        subjects.append('s' + str(i).zfill(2))

    subjects.remove('s29')
    subjects.remove('s34')
    subL = ['s', 'se']

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)
    chans = [4, 12, 20, 47,30, 39, 49, 57, 6, 14, 22, 26, 41, 51, 59, 63]
    for i, sub in enumerate(subjects):
        rawData = loadmat(os.path.join(datasetPath, sub + '.mat'))['eeg']
        rawDataLeft = rawData[0][0][7][:64, :]
        rawDataRight = rawData[0][0][8][:64, :]
        rawOnesetPoint = np.where(rawData[0][0][11] > 0)[1]

        rawDataLeft = rawDataLeft * 1e-6
        rawDataRight = rawDataRight * 1e-6
        rawDataLeft = exponential_moving_standardize(rawDataLeft, 1e-3)
        rawDataRight = exponential_moving_standardize(rawDataRight, 1e-3)

        offset = 0
        fs = 512
        epochWindow = [0, 3]
        epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + int(offset * fs)
        x_left = np.stack([rawDataLeft[:, epochInterval + time] for time in rawOnesetPoint], axis=2)
        x_right = np.stack([rawDataRight[:, epochInterval + time] for time in rawOnesetPoint], axis=2)

        xNewl = np.zeros((x_left.shape[0], 750, x_left.shape[2]), np.float32)
        xNewr = np.zeros((x_right.shape[0], 750, x_right.shape[2]), np.float32)
        for j in range(x_left.shape[0]):  # resampy.resample cant handle the 3D data.
            xNewl[j, :, :] = resampy.resample(x_left[j, :, :], 512, 250, axis=0)
            xNewr[j, :, :] = resampy.resample(x_right[j, :, :], 512, 250, axis=0)
        x_left = xNewl
        x_right = xNewr

        x_left = x_left.transpose((2, 0, 1))
        x_right = x_right.transpose((2, 0, 1))
        x = np.concatenate([x_left, x_right], axis=0)
        # x = np.clip(x, -800, 800)

        len = x.shape[0]
        leftLabel = np.ones(len // 2, dtype=int) - 1
        rightLabel = np.ones(len // 2, dtype=int)
        label = np.concatenate([leftLabel, rightLabel], axis=0)

        index = [i for i in range(len)]
        random.seed(6)
        random.shuffle(index)

        x = x[index]
        label = label[index]

        x = alignOperation(x, operation='svd')
        # x, y = online_operation(x, label, start=2, n=2)

        dataS1 = {'x': x.transpose((1, 2, 0)), 'y': y, 'c': 64, 's': 512}
        # dataS2 = {'x': x[len // 2:].transpose((1, 2, 0)), 'y': label[len // 2:], 'c': 64, 's': 512}
        savemat(os.path.join(savePath, 's' + str(i + 1).zfill(3) + '.mat'), dataS1)
        # savemat(os.path.join(savePath, 'se' + str(i + 1).zfill(3) + '.mat'), dataS2)
    return


def parseBci32aDataset(datasetPath, savePath, ):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    channel = ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1',
               'FCz', 'FC2', 'FC4', 'FC6', 'T7', 'T8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
               'TP7', 'TP8', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3',
               'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']
    subject = ['aa', 'al', 'av', 'aw', 'ay']
    subject_train_len = [168, 224, 84, 56, 28]
    for i, sub in enumerate(subject):
        rawData = loadmat(os.path.join(datasetPath, 'data_set_IVa_' + sub + '.mat'))
        index = []
        for j in range(len(rawData['nfo'][0][0][2][0])):
            if rawData['nfo'][0][0][2][0][j][0] in channel:
                index.append(j)
        data = rawData['cnt'].transpose((1, 0))[index]
        data = data * 0.1
        # data = exponential_moving_standardize(data, 1e-3)

        rawLabel = loadmat(os.path.join(datasetPath, 'true_labels_' + sub + '.mat'))
        label = np.array(rawLabel['true_y'][0] - 1, dtype=int)

        offset = 0.5
        fs = 100
        epochWindow = [0, 3]
        epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + int(offset * fs)

        data_ = np.stack([data[:, epochInterval + time] for time in rawData['mrk'][0][0][0][0]], axis=2)

        trainLen = subject_train_len[i]

        saveData_train = {'x': data_[:, :, :trainLen], 'y': label[:trainLen], 'c': channel, 's': fs}
        saveData_test = {'x': data_[:, :, trainLen:], 'y': label[trainLen:], 'c': channel, 's': fs}

        savemat(os.path.join(savePath, 's' + str(i + 1).zfill(3) + '.mat'), saveData_train)
        savemat(os.path.join(savePath, 'se' + str(i + 1).zfill(3) + '.mat'), saveData_test)
    return


def parsePhysionetDataset(datasetPath, savePath, cls=False):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    leftRight = [4, 8, 12]
    fistFeet = [6, 10, 14]
    eventCode = [2, 3]
    epochWindow = [0, 3]
    fs = 160
    offset = 0
    chans = [8, 10, 12, 31, 35, 48, 50, 52, 29, 40, 46, 60, 37, 41, 54, 62]

    for i in range(1, 110):
        sub = os.path.join(datasetPath, 'S' + str(i).zfill(3))
        data = []
        label = []
        for j in leftRight:
            dataPath = os.path.join(sub, 'S' + str(i).zfill(3) + 'R' + str(j).zfill(2) + '.edf')
            raw = mne.io.read_raw_edf(dataPath, stim_channel="auto")
            raw.load_data()
            gdf_events = mne.events_from_annotations(raw)[0][:, [0, 2]].tolist()
            eeg = raw.get_data()
            # eeg = eeg[chans, :]
            events = [event for event in gdf_events if event[1] in eventCode]
            y = np.array([i[1] for i in events])
            epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
            x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)
            x = x * 1e6
            # left 0, right 1
            y = y - 2
            data.append(x)
            label.append(y)

        if cls:
            for j in fistFeet:
                dataPath = os.path.join(sub, 'S' + str(i).zfill(3) + 'R' + str(j).zfill(2) + '.edf')
                raw = mne.io.read_raw_edf(dataPath, stim_channel="auto")
                raw.load_data()
                gdf_events = mne.events_from_annotations(raw)[0][:, [0, 2]].tolist()
                eeg = raw.get_data()
                # eeg = eeg[chans, :]
                events = [event for event in gdf_events if event[1] in eventCode]
                y = np.array([i[1] for i in events])
                epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
                x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)
                x = x * 1e6
                # fist 2, feet 3
                y = y
                data.append(x)
                label.append(y)

        for d in range(len(data)):
            if d == 0:
                data_ = data[0]
                label_ = label[0]
            else:
                data_ = np.concatenate([data_, data[d]], axis=2)
                label_ = np.concatenate([label_, label[d]], axis=0)

        data_ = data_.transpose((2, 0, 1))
        data_ = alignOperation(data_, operation='svd')
        data_ = data_.transpose((1, 2, 0))

        saveData_train = {'x': data_, 'y': label_, 'c': np.array(raw.info['ch_names'])[:].tolist(), 's': fs}

        savemat(os.path.join(savePath, 's' + str(i).zfill(3) + '.mat'), saveData_train)

    return


def parseSHUDataset(datasetPath, savePath):
    fs = 250
    ch_names = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
                "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
                "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
                "O1", "O2"]

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for i in range(1, 26):
        for j in range(1, 6):
            sub = os.path.join(datasetPath, 'sub-'+str(i).zfill(3)+'_ses-'+str(j).zfill(2)+'_task_motorimagery_eeg.mat')
            mat = loadmat(sub)
            x = mat['data']
            y = np.ravel(mat['labels']) - 1

            x = alignOperation(x, operation='svd')
            # x, y = online_operation(x, y, start=0, n=1)
            x = x.transpose((1, 2, 0))

            saveData_train = {'x': x, 'y': y, 'c': np.array(ch_names)[:].tolist(), 's': fs}

            savemat(os.path.join(savePath, 's-s' + str(j) + '-' + str(i).zfill(3) + '.mat'), saveData_train)

    return


def fetchAndParseKoreaFile(dataPath, url=None, epochWindow=[0, 4],
                           chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],
                           downsampleFactor=4):
    eventCode = [1, 2]  # start of the trial at t=0
    s = 1000
    offset = 0

    # check if the file exists or fetch it over ftp
    if not os.path.exists(dataPath):
        if not os.path.exists(os.path.dirname(dataPath)):
            os.makedirs(os.path.dirname(dataPath))
        print('fetching data over ftp: ' + dataPath)
        with closing(request.urlopen(url)) as r:
            with open(dataPath, 'wb') as f:
                shutil.copyfileobj(r, f)

    # read the mat file:
    try:
        data = loadmat(dataPath)
    except:
        print('Failed to load the data. retrying the download')
        with closing(request.urlopen(url)) as r:
            with open(dataPath, 'wb') as f:
                shutil.copyfileobj(r, f)
        data = loadmat(dataPath)

    x = np.concatenate((data['EEG_MI_train'][0, 0]['smt'], data['EEG_MI_test'][0, 0]['smt']), axis=1).astype(np.float32)
    y = np.concatenate((data['EEG_MI_train'][0, 0]['y_dec'].squeeze(), data['EEG_MI_test'][0, 0]['y_dec'].squeeze()),
                       axis=0).astype(int) - 1
    c = np.array([m.item() for m in data['EEG_MI_train'][0, 0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0, 0]['fs'].squeeze().item()
    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:, :, np.array(chans)]
        c = c[np.array(chans)]

    # down-sample if requested .
    if downsampleFactor is not None:
        xNew = np.zeros((int(x.shape[0] / downsampleFactor), x.shape[1], x.shape[2]), np.float32)
        for i in range(x.shape[2]):  # resampy.resample cant handle the 3D data.
            xNew[:, :, i] = resampy.resample(x[:, :, i], s, s / downsampleFactor, axis=0)
        x = xNew
        s = s / downsampleFactor

    x = x.transpose((1, 2, 0))
    x = alignOperation(x, operation='svd')
    x = x.transpose((1, 2, 0))

    # change the data dimensions to be in a format: Chan x time x trials
    # x = np.transpose(x, axes=(2, 0, 1))

    return {'x': x, 'y': y, 'c': c, 's': s}


def parseKoreaDataset(datasetPath, savePath, epochWindow=[0, 4],
                      chans=[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20],
                      downsampleFactor=4, verbos=False):
    chans = list(range(62))
    # Base url for fetching any data that is not present!
    fetchUrlBase = 'ftp://parrot.genomics.cn/gigadb/pub/10.5524/100001_101000/100542/'
    subjects = list(range(54))
    subAll = [subjects, subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))
            if not os.path.exists(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat')):
                fileUrl = fetchUrlBase + 'session' + str(iSubs + 1) + '/' + 's' + str(iSub + 1) + '/' + 'sess' + str(
                    iSubs + 1).zfill(2) + '_' + 'subj' + str(iSub + 1).zfill(2) + '_EEG_MI.mat'
            data = fetchAndParseKoreaFile(
                os.path.join(datasetPath, 'session' + str(iSubs + 1), 's' + str(iSub + 1), 'EEG_MI.mat'),
                None, epochWindow=epochWindow, chans=chans, downsampleFactor=downsampleFactor)

            savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)


def matToPython(datasetPath, savePath, isFiltered=False):
    print('Creating python eegdataset with raw data ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # load all the mat files
    data = [];
    for root, dirs, files in os.walk(datasetPath):
        files = sorted(files)
        for f in files:
            parD = {}
            parD['fileName'] = f
            parD['data'] = {}
            d = loadmat(os.path.join(root, f),
                        verify_compressed_data_integrity=False)
            if isFiltered:
                parD['data']['eeg'] = np.transpose(d['x'], (2, 0, 1, 3)).astype('float32')
            else:
                parD['data']['eeg'] = np.transpose(d['x'], (2, 0, 1)).astype('float32')

            parD['data']['labels'] = d['y']
            data.append(parD)

    id = 0
    dataLabels = [['id', 'relativeFilePath', 'label', 'subject', 'session']]  # header row
    for i, d in enumerate(data):

        sub = int(d['fileName'][-7:-4])  # subject of the data
        sub = str(sub).zfill(3)

        if d['fileName'][1] == 'e':
            session = 1;
        elif d['fileName'][1] == '-':
            session = int(d['fileName'][3:4])
        else:
            session = 0;

        if len(d['data']['labels']) == 1:
            d['data']['labels'] = np.transpose(d['data']['labels'])
        # for j, label in enumerate(d['data']['labels']):
        #     print(j,label)
        for j, label in enumerate(d['data']['labels']):
            lab = label[0]
            # get the data
            if isFiltered:
                # x = {'id': id, 'data': d['data']['eeg'][j, :, :, :], 'label': lab}
                x = [d['data']['eeg'][j, :, :, :], lab]
            else:
                # x = {'id': id, 'data': d['data']['eeg'][j, :, :], 'label': lab}
                x = [d['data']['eeg'][j, :, :], lab]

            # dump it in the folder
            with open(os.path.join(savePath, str(id).zfill(5) + '.dat'), 'wb') as fp:
                pickle.dump(x, fp)

            # add in data label file
            dataLabels.append([id, str(id).zfill(5) + '.dat', lab, sub, session])

            # increment id
            id += 1
    # Write the dataLabels file as csv
    with open(os.path.join(savePath, "dataLabels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataLabels)

    # write miscellaneous data info as csv file
    dataInfo = [['fs', 250], ['chanName', 'Check Original File']]
    with open(os.path.join(savePath, "dataInfo.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataInfo)


def pythonToMultiviewPython(datasetPath, savePath,
                            filterTransform=None):
    # 此处加入了 filter trans form
    trasnformAndSave(datasetPath, savePath, transform=filterTransform)


def trasnformAndSave(datasetPath, savePath, transform=None):
    if transform is None:
        return -1

    # Data options:
    config = {}
    config['preloadData'] = False  # process One by one
    config['transformArguments'] = transform
    config['inDataPath'] = datasetPath
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Outputs will be saved in folder : ' + savePath)

    # Check and compose transforms
    if len(config['transformArguments']) > 1:
        transform = transforms.Compose(
            [transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
    else:
        transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](
            **config['transformArguments'][list(config['transformArguments'].keys())[0]])

    data = eegDataset(dataPath=config['inDataPath'], dataLabelsPath=config['inLabelPath'],
                      preloadData=config['preloadData'], transform=transform)

    # Write the transform applied data
    dLen = len(data)
    perDone = 0

    for i, d in enumerate(data):
        with open(os.path.join(savePath, data.labels[i][1]), 'wb') as fp:  # 1-> realtive-path
            pickle.dump(d, fp)
        if i / dLen * 100 > perDone:
            print(str(perDone) + '% Completed')
            perDone += 1

    # Copy the labels and config files
    copyfile(config['inLabelPath'], os.path.join(savePath, 'dataLabels.csv'))
    copyfile(os.path.join(config['inDataPath'], "dataInfo.csv"), os.path.join(savePath, "dataInfo.csv"))

    # Store the applied transform in the transform . csv file
    with open(os.path.join(config['inDataPath'], "transform.csv"), 'w') as f:
        for key in config['transformArguments'].keys():
            f.write("%s,%s\n" % (key, config['transformArguments'][key]))


def fetchData(dataFolder, datasetId=0, filterTransform=None):
    print('fetch ssettins: ', dataFolder, datasetId)
    oDataFolder = 'originalData'
    rawMatFolder = 'rawMat'
    rawPythonFolder = 'rawPython'
    multiviewPythonFolder = 'multiviewPython'

    # check that all original data exists
    if not os.path.exists(os.path.join(dataFolder, oDataFolder)):
        ValueError('The original dataset doesn\'t exist at path: ' +
                   os.path.join(dataFolder, oDataFolder) +
                   ' Please download and copy the extracted dataset at the above path ' +
                   ' More details about how to download this data can be found in the instructions.txt file')
    else:
        oDataLen = len([name for name in os.listdir(os.path.join(dataFolder, oDataFolder))
                        if os.path.isfile(os.path.join(dataFolder, oDataFolder, name))])
        if datasetId == 0:
            pass
        elif datasetId == 1 and oDataLen < 108:
            pass
        elif datasetId == 2:
            pass
        elif datasetId == 3:
            pass
        elif datasetId == 4:
            pass
        elif datasetId == 5:
            pass

    # Check if the processed mat data exists:
    # 看mat创建mt
    if not os.path.exists(os.path.join(dataFolder, rawMatFolder)):
        print('Appears that the raw data exists but its not parsed yet. Starting the data parsing ')
        if datasetId == 0:
            parseBci42aDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 1:
            parseKoreaDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 2:
            parseBci42bDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 3:
            parseHGDDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 4:
            parseGISTDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 5:
            parseBci32aDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 6:
            parsePhysionetDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder), True)
        elif datasetId == 7:
            parsePhysionetDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder), False)
        elif datasetId == 8:
            parseSHUDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))

    # Check if the processed python data exists:
    if not os.path.exists(os.path.join(dataFolder, rawPythonFolder, 'dataLabels.csv')):
        print(
            'Appears that the parsed mat data exists but its not converted to eegdataset yet. Starting this processing')
        matToPython(os.path.join(dataFolder, rawMatFolder), os.path.join(dataFolder, rawPythonFolder))

    # Check if the multi-view python data exists:
    if not os.path.exists(os.path.join(dataFolder, multiviewPythonFolder, 'dataLabels.csv')):
        print(
            'Appears that the raw eegdataset data exists but its not converted to multi-view eegdataset yet. Starting this processing')
        pythonToMultiviewPython(os.path.join(dataFolder, rawPythonFolder),
                                os.path.join(dataFolder, multiviewPythonFolder),
                                filterTransform=filterTransform)

    print('All the data you need is present! ')
    return 1


def exponential_moving_standardize(data, factor_new=0.001, init_block_size=None, eps=1e-4):
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T


def alignOperation(x, operation=None):
    # x.shape (b, c, t)
    if operation == 'ea':
        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        r_op = inv(sqrtm(r))
        if np.iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. Can be caused by running ICA-EOG rejection, if "
                  "not, check data!!")
            r_op = np.real(r_op).astype(np.float32)
        elif not np.any(np.isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")
        x = np.matmul(r_op, x)
    elif operation == 'zca':
        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        d, v = np.linalg.eigh(r)
        w = np.diag(1. / np.sqrt(d + 1e-18))
        w = w.real.round(4)
        wpca = np.matmul(w, v.T)
        wzca = np.matmul(np.matmul(v, w), v.T)
        x = np.matmul(wzca, x)
    elif operation == 'svd':
        r = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
        U, S, V = np.linalg.svd(r)
        w = np.diag(1.0 / np.sqrt(S + 1e-18))
        wsvd = np.matmul(U, np.matmul(w, U.T))
        x = np.matmul(wsvd, x)
    return x


def online_operation(x, y, start=5, n=100):
    tmp = np.zeros([x.shape[0] - start, x.shape[1], x.shape[2]])
    y = y[start:]
    for i in range(tmp.shape[0]):
        if i > n:
            tmp[i] = alignOperation(x[i + start + 1 - n: i+start+1], operation='svd')[-1]
        else:
            tmp[i] = alignOperation(x[:i+start+1], operation='svd')[-1]
    return tmp, y
