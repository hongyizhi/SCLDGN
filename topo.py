#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import sys
import os
import time
import xlwt
import random
import math
import copy

masterPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(masterPath, ''))
from dataset.eegDataset import eegDataset
from baseModel.baseModel import baseModel
from network import networks
from network.SMAA import ERM_SMA
from network.gradCam import GradCAM
from dataset.saveData import fetchData
from utils.tools import setRandom, excelAddData, dictToCsv, count_parameters, get_transform, \
    getModelArguments, ConcatDatasetWithDomainLabel, getBaseModelArguments, split_dataset, split_list, split_idx
import mne
import matplotlib
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# reporting settings
debug = False


def ho(datasetId=None, network=None, numEpochs=200, maxEpochs=200, batchSize=32, feature=32, subTorun=None, ps='',
       targetSize=32, dropoutP=0., filterType='cheby2', c=0.5, tradeOff=0, tradeOff2=0., tradeOff3=100, tradeOff4=10,
       isProj=True, draw=True, algorithm='ce', typetype='scldgn'):
    datasets = ['bci42a', 'korea', 'bci42b', 'hgd', 'gist', 'bci32a', 'physionet', 'physionet2']
    config = {}
    config['randSeed'] = 19960822
    config['preloadData'] = False
    config['network'] = network
    config['modelArguments'] = getModelArguments(datasetId=datasetId, dropoutP=dropoutP, feature=feature,
                                                 c=c, isProj=isProj)
    config['baseModelArugments'] = getBaseModelArguments(datasetId=datasetId, batchSize=batchSize,
                                                         targetSize=targetSize, tradeOff=tradeOff,
                                                         tradeOff2=tradeOff2, tradeOff3=tradeOff3,
                                                         tradeOff4=tradeOff4, algorithm=algorithm)

    # Training related details
    config['modelTrainArguments'] = {
        'stopCondi': {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': maxEpochs, 'varName': 'epoch'}},
                                   'c2': {'NoDecrease': {'numEpochs': numEpochs, 'varName': 'valInacc'}}}}},
        'sampler': 'RandomSampler', 'loadBestModel': True, 'bestVarToCheck': 'valInacc', 'continueAfterEarlystop': True,
        'lr': 1e-3}

    # modeInFol = 'multiviewPython'
    # config['inDataPath'] = os.path.join(masterPath, '..', 'data', datasets[datasetId], modeInFol)
    # config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    if typetype == 'eegnet':
        viewType = 'multiviewPythonraw1'
        viewType_nofb = 'multiviewPythonraw1'
    elif typetype == 'baseline':
        viewType = 'multiviewPythonraw9'
        viewType_nofb = 'multiviewPythonraw1'
    else:
        viewType = 'multiviewPython'
        viewType_nofb = 'multiviewPython2'

    config['inDataPath'] = r'E:\research\data'
    config['inDataPath'] = os.path.join(config['inDataPath'], datasets[datasetId], viewType)
    # config['inDataPath'] = r'E:\research\CrossNet\data\bci42a\multiviewPython'
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    config['outPath'] = os.path.join(masterPath, 'output', datasets[datasetId], 'dg')
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())) + network + ps
    config['outPath'] = os.path.join(config['outPath'], randomFolder, '')

    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])

    #  filterType: butter, cheby2, fir, none
    config['transformArguments'] = get_transform(filtBank=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24],
                                                           [24, 28], [28, 32], [32, 36], [36, 40]],
                                                 fs=250, filterType=filterType, order=3, filtType='filter',
                                                 outputType='sos')

    # %% check and Losad the data
    print('Data loading in progress')
    fetchData(os.path.dirname(config['inDataPath']), datasetId, filterTransform=config['transformArguments'])
    data = eegDataset(dataPath=config['inDataPath'], dataLabelsPath=config['inLabelPath'],
                      preloadData=config['preloadData'], transform=None)

    NoFBPath = r'E:\research\data'
    NoFBPath = os.path.join(NoFBPath, datasets[datasetId], viewType_nofb)
    NoFBPath1 = os.path.join(NoFBPath, 'dataLabels.csv')
    dataNoFB = eegDataset(dataPath=NoFBPath, dataLabelsPath=NoFBPath1,
                          preloadData=config['preloadData'], transform=None)

    print('Data loading finished')

    # import networks
    if config['network'] in networks.__dict__.keys():
        network = networks.__dict__[config['network']]
    else:
        raise AssertionError('No network named ' + config['network'] + ' is not defined in the networks.py file')

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    config['ps'] = ps
    config['numEpochs'] = numEpochs
    config['feature'] = feature
    config['subTorun'] = subTorun
    config['split_ratio'] = 0.8
    config['parameters'] = str(count_parameters(net))
    config['net'] = net
    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

    setRandom(config['randSeed'])
    net = network(**config['modelArguments'])
    netInitState = net.to('cpu').state_dict()
    # torch.save(netInitState, os.path.join(masterPath, 'savepoint.pth'))

    # %% Find all the subjects to run
    subs = sorted(set([d[3] for d in data.labels]))
    nSub = len(subs)

    if config['subTorun']:
        config['subTorun'] = list(range(config['subTorun'][0], config['subTorun'][1]))
    else:
        config['subTorun'] = list(range(nSub))

    def handle(data, label):
        dataCam = torch.unsqueeze(data.data[0][0], dim=0)
        lent = len(data.data)
        for i in range(0, lent - 1):
            # if data.data[i + 1][1] == 0 or data.data[i + 1][1] == 1:
            if data.data[i + 1][1] == label:
                dataCam1 = torch.unsqueeze(data.data[i + 1][0], dim=0)
                dataCam = torch.cat((dataCam, dataCam1), dim=0)
        return dataCam

    def cal1(raw, map, i, tr=1):
        # np.min() np.max()
        mean_trial = np.mean(raw[i:i + 1], axis=0)
        test_all_cam = np.mean(map[i:i + 1], axis=0)  # \in [0, 1]

        mean_trial_ = (mean_trial - np.mean(mean_trial)) / (np.max(mean_trial) - np.min(mean_trial))
        test_all_cam_ = (test_all_cam - np.mean(test_all_cam)) / (np.max(test_all_cam) - np.min(test_all_cam))

        hyb_all = mean_trial_ * test_all_cam_
        m_hyb_all = np.mean(hyb_all, axis=1)
        return m_hyb_all

        # if tr == 1:
        #     cam = np.mean(test_all_cam_, axis=1)
        #     return cam
        # elif tr == 2:
        #     hyb_all = mean_trial_ * test_all_cam_
        #     m_hyb_all = np.mean(hyb_all, axis=1)
        #     return m_hyb_all
        # elif tr == 3:
        #     hyb_all_sum = mean_trial_ + test_all_cam_
        #     s_hyb_all = np.mean(hyb_all_sum, axis=1)
        #     return s_hyb_all

    if datasetId == 0:
        biosemi_montage = mne.channels.make_standard_montage('biosemi64')
        index = [37, 9, 10, 46, 45, 44, 13, 12, 11, 47, 48, 49, 50, 17, 18, 31, 55, 54, 19, 30, 56,
                 29]  # for bci competition iv 2a
        biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
        biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
        info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

        mean = np.zeros((22, 1000))
        evoked = mne.EvokedArray(mean, info)
        evoked.set_montage(biosemi_montage)
        sample_len = 140

    elif datasetId == 4:
        biosemi_montage = mne.channels.make_standard_montage('biosemi64')
        info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')
        mean = np.zeros((64, 750))
        evoked = mne.EvokedArray(mean, info)
        evoked.set_montage(biosemi_montage)
        sample_len = 100

    elif datasetId == 7:
        biosemi_montage = mne.channels.make_standard_montage('biosemi64')
        info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=160., ch_types='eeg')
        mean = np.zeros((64, 480))
        evoked = mne.EvokedArray(mean, info)
        evoked.set_montage(biosemi_montage)
        sample_len = 17

    elif datasetId == 3:
        biosemi_montage = mne.channels.make_standard_montage('standard_1005')
        index = [27, 29, 31, 33, 39, 43, 49, 51, 53, 55, 28, 30, 32, 38, 40, 42, 44, 50, 52, 54, 108,
             109, 112, 113, 118, 119, 122, 123, 128, 129, 132, 133, 138, 139, 142,143, 110, 111,
             120, 121, 130, 131, 140, 141]
        biosemi_montage.ch_names = [biosemi_montage.ch_names[i] for i in index]
        biosemi_montage.dig = [biosemi_montage.dig[i + 3] for i in index]
        info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=250., ch_types='eeg')

        mean = np.zeros((44, 1000))
        evoked = mne.EvokedArray(mean, info)
        evoked.set_montage(biosemi_montage)
        sample_len = 140


    for iSub, sub in enumerate(subs):

        if iSub not in config['subTorun']:
            continue

        pth_path = r'E:\manuscript\SMCLDGN\data\topo'
        if typetype == 'scldgn':
            pth_path = os.path.join(pth_path, datasets[datasetId], 'pth')
        else:
            pth_path = os.path.join(pth_path, datasets[datasetId], typetype)
        pth_path = pth_path + '\\' + 'sub' + str(iSub) + '%' + '5Ccheckpoint.pth.tar'
        task = [str(i).zfill(3) for i in range(1, config['baseModelArugments']['ndomain'] + 1)]

        TrainIdxList = []

        # 测试数据
        subIdx_test = [i for i, x in enumerate(data.labels) if x[3] == sub]
        # subIdx_test = [i for i, x in enumerate(data.labels)]
        test_len = len(subIdx_test)

        subDataNoFB = copy.deepcopy(dataNoFB)
        subDataNoFB.createPartialDataset(subIdx_test, loadNonLoadedData=True)

        testData = copy.deepcopy(data)
        testData.createPartialDataset(subIdx_test, loadNonLoadedData=True)

        setRandom(config['randSeed'])

        checkpoint = torch.load(pth_path)
        net_ = ERM_SMA(net, start=tradeOff3)
        net_.load_state_dict(checkpoint['state_dict'], strict=False)

        dataCam0 = handle(testData, 0)
        dataCam1 = handle(testData, 1)

        dataCamNoFB0 = handle(subDataNoFB, 0)
        dataCamNoFB0 = dataCamNoFB0.numpy()
        dataCamNoFB0 = dataCamNoFB0[:, :, :, 0]

        dataCamNoFB1 = handle(subDataNoFB, 1)
        dataCamNoFB1 = dataCamNoFB1.numpy()
        dataCamNoFB1 = dataCamNoFB1[:, :, :, 0]

        # target_layers = [net_.network_sma.feature[6], net_.network_sma.feature[7], net_.network_sma.feature[8]]
        if typetype == 'eegnet':
            layer = net_.network_sma.firstBlocks[0][2]
        else:
            layer = net_.network_sma.feature[6]

        target_layers = [layer]

        if tradeOff == 1:
            cam = GradCAM(model=net_.network_sma, target_layers=target_layers, use_cuda=True)
        elif tradeOff == 2:
            cam = GradCAMPlusPlus(model=net_.network_sma, target_layers=target_layers, use_cuda=True)
        elif tradeOff == 3:
            cam = LayerCAM(model=net_.network_sma, target_layers=target_layers, use_cuda=True)

        all_cam0 = []
        all_cam1 = []
        datac0 = dataCam0.permute(0, 3, 1, 2)
        datac1 = dataCam1.permute(0, 3, 1, 2)
        for i in range(datac0.size(0)):
            # test = data[i:i + 1, :, :, :]
            target = [ClassifierOutputTarget(0)]
            test = torch.as_tensor(datac0[i:i + 1, :, :, :], dtype=torch.float32)
            test = torch.autograd.Variable(test, requires_grad=True)
            grayscale_cam = cam(input_tensor=test, targets=target)
            grayscale_cam = grayscale_cam[0, :]
            all_cam0.append(grayscale_cam)

        for i in range(datac1.size(0)):
            # test = data[i:i + 1, :, :, :]
            target = [ClassifierOutputTarget(1)]
            test = torch.as_tensor(datac1[i:i + 1, :, :, :], dtype=torch.float32)
            test = torch.autograd.Variable(test, requires_grad=True)
            grayscale_cam = cam(input_tensor=test, targets=target)
            grayscale_cam = grayscale_cam[0, :]
            all_cam1.append(grayscale_cam)

        # all = []
        for j in range(1, min(datac0.size(0), datac1.size(0))):
            all0 = []
            all1 = []
            for i in range(j, j + 1):
                mean_hyb_all0 = cal1(dataCamNoFB0, all_cam0, i, tr=tradeOff)
                mean_hyb_all1 = cal1(dataCamNoFB1, all_cam1, i, tr=tradeOff)
                all0.append(mean_hyb_all0)
                all1.append(mean_hyb_all1)

            all0 = np.mean(all0, axis=0)
            all1 = np.mean(all1, axis=0)

            if datasetId == 7:
                index = [21, 24, 25, 32, 31, 30, 29, 38, 0, 1, 2, 9, 8, 7, 40, 44, 14, 15, 16, 49, 48, 47, 46, 42,
                         55, 56, 60, 63, 61, 57, 50, 17, 22, 23, 28, 27, 26, 33, 34, 35, 36, 37, 39, 6, 5,
                         4, 3, 10, 11, 12, 13, 41, 45, 20, 19, 18, 51, 52, 53, 54, 43, 59, 58, 62]
                all0 = all0[index]
                all1 = all1[index]

            save_pth = r'E:\manuscript\SMCLDGN\data\topo\topo_all'
            save_pth = os.path.join(save_pth, datasets[datasetId], typetype)

            np.save(os.path.join(save_pth, 'sub%s_%s_left.npy' % (sub, j)), all0)
            np.save(os.path.join(save_pth, 'sub%s_%s_right.npy' % (sub, j)), all1)

            if draw:
                plt.figure()

                ax1 = plt.subplot(211)
                im1, cn1 = mne.viz.plot_topomap(all0, evoked.info, show=False, size=1, axes=ax1, res=300, )

                ax2 = plt.subplot(212)
                im2, cn2 = mne.viz.plot_topomap(all1, evoked.info, show=False, size=1, axes=ax2, res=300, )

                a = plt.colorbar(im1, shrink=1)
                a.ax.tick_params(size=0)
                a.ax.set_yticks([])

                b = plt.colorbar(im2, shrink=1)
                b.ax.tick_params(size=0)
                b.ax.set_yticks([])
                plt.tight_layout()

                fig_path = r'E:\manuscript\SMCLDGN\data\topo'
                if typetype == 'scldgn':
                    if tradeOff == 1:
                        folder = 'gradcam'
                    elif tradeOff == 2:
                        folder = 'gradcamplus'
                    elif tradeOff == 3:
                        folder = 'layercam'
                else:
                    folder = typetype
                fig_path = os.path.join(fig_path, datasets[datasetId], folder)
                fig_path = os.path.join(fig_path, 'sub%s_%s.png' % (sub, j))

                plt.savefig(fig_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # algorithms ['ce', 'coral', 'scl', 'smcldgn', 'smcldgn_mc','mixup', 'mmd', 'dann', 'irm', 'mldg']

    ho(datasetId=0, network='B7', batchSize=32, feature=32, subTorun=[6, 7], isProj=False,
       tradeOff=1, tradeOff2=0, maxEpochs=1, algorithm='smcldgn_mc', draw=True, typetype='scldgn')
