#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import os
import time
import copy

masterPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(masterPath, ''))
from dataset.eegDataset import eegDataset
from baseModel.baseModel import baseModel
from network import networks
from network.SMA import ERM_SMA
from dataset.saveData import fetchData
from utils.tools import setRandom, dictToCsv, count_parameters, get_transform, \
    getModelArguments, ConcatDatasetWithDomainLabel, getBaseModelArguments, split_idx, save_book_simple, save_book

# reporting settings
debug = False


def ho(datasetId=None, network=None, numEpochs=200, maxEpochs=200, batchSize=32, feature=32, subTorun=None, ps='',
       dropoutP=0., filterType='cheby2', c=0.5, tradeOff=0., tradeOff2=0., tradeOff3=0., tradeOff4=10, sma=100,
       isProj=True, algorithm='ce'):
    datasets = ['bci42a', 'korea', 'bci42b', 'hgd', 'gist', 'bci32a', 'physionet', 'physionet2']
    config = {}
    config['randSeed'] = 19960822
    config['preloadData'] = False
    config['network'] = network
    config['modelArguments'] = getModelArguments(datasetId=datasetId, dropoutP=dropoutP, feature=feature,
                                                 c=c, isProj=isProj)
    config['baseModelArugments'] = getBaseModelArguments(datasetId=datasetId, batchSize=batchSize,
                                                         tradeOff=tradeOff,
                                                         tradeOff2=tradeOff2, tradeOff3=tradeOff3,
                                                         tradeOff4=tradeOff4, algorithm=algorithm)

    # Training related details
    config['modelTrainArguments'] = {
        'stopCondi': {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': maxEpochs, 'varName': 'epoch'}},
                                   'c2': {'NoDecrease': {'numEpochs': numEpochs, 'varName': 'valInacc'}}}}},
        'sampler': 'RandomSampler', 'loadBestModel': True, 'bestVarToCheck': 'valInacc', 'lr': 1e-3}

    modeInFol = 'multiviewPython'
    config['inDataPath'] = os.path.join(masterPath, '..', 'data', datasets[datasetId], modeInFol)
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    config['outPath'] = os.path.join(masterPath, 'output', datasets[datasetId])
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
    print('Data loading finished')

    # import networks
    if config['network'] in networks.__dict__.keys():
        network = networks.__dict__[config['network']]
    else:
        raise AssertionError('No network named ' + config['network'] + ' is not defined in the networks.py file')

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    config['subTorun'] = subTorun
    config['split_ratio'] = 0.8
    config['parameters'] = str(count_parameters(net))
    config['net'] = net
    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'], 'config.csv'), config)

    setRandom(config['randSeed'])
    net = network(**config['modelArguments'])
    netInitState = net.to('cpu').state_dict()

    # %% Find all the subjects to run
    subs = sorted(set([d[3] for d in data.labels]))
    nSub = len(subs)

    if config['subTorun']:
        config['subTorun'] = list(range(config['subTorun'][0], config['subTorun'][1]))
    else:
        config['subTorun'] = list(range(nSub))

    # %% Let the training begin
    trainResults = []
    valResults = []
    testResults = []
    bestEpoch = []
    totalTime = []
    loadingDataTime = {}
    start = time.time()
    ifLoad = True
    for iSub, sub in enumerate(subs):

        if iSub not in config['subTorun']:
            continue

        task = [str(i).zfill(3) for i in range(1, config['baseModelArugments']['ndomain'] + 1)]
        task.remove(sub)

        TrainIdxList = []
        ValIdxList = []
        # 处理数据
        for i, t in enumerate(task):
            TmpList = []
            TrainTmp = []
            ValTmp = []
            TmpList.append([j for j, d in enumerate(data.labels) if d[3] in t and 0 == d[2]])
            TmpList.append([j for j, d in enumerate(data.labels) if d[3] in t and 1 == d[2]])
            TmpList.append([j for j, d in enumerate(data.labels) if d[3] in t and 2 == d[2]])
            TmpList.append([j for j, d in enumerate(data.labels) if d[3] in t and 3 == d[2]])

            for eachTask in TmpList:
                idx1, idx2 = split_idx(eachTask, config['split_ratio'], config['randSeed'], datasetId)
                TrainTmp += [item for item in idx1]
                ValTmp += [item for item in idx2]
            TrainIdxList.append(TrainTmp)
            ValIdxList.append(ValTmp)

        taskDatasetList = []
        taskValDatasetList = []
        taskTestDatasetList = []
        for i in range(len(TrainIdxList)):
            trainData = copy.deepcopy(data)
            trainData.createPartialDataset(TrainIdxList[i], loadNonLoadedData=True)
            valData = copy.deepcopy(data)
            valData.createPartialDataset(ValIdxList[i], loadNonLoadedData=True)
            taskDatasetList.append(trainData)
            taskValDatasetList.append(valData)

        trainData = ConcatDatasetWithDomainLabel(taskDatasetList)
        valData = ConcatDatasetWithDomainLabel(taskValDatasetList)

        # 测试数据
        subIdx_test = [i for i, x in enumerate(data.labels) if x[3] in sub]

        testData = copy.deepcopy(data)
        testData.createPartialDataset(subIdx_test, loadNonLoadedData=True)
        taskTestDatasetList.append(testData)
        testData = ConcatDatasetWithDomainLabel(taskTestDatasetList)

        loadingDataTime['train data number'] = trainData.__len__()
        loadingDataTime['val   data number'] = valData.__len__()
        loadingDataTime['test  data number'] = testData.__len__()
        loadingDataTime['lodinging time'] = np.round((time.time() - start) / 60, 2)
        if ifLoad:
            dictToCsv(os.path.join(config['outPath'], 'loadingTime.csv'), loadingDataTime)
            ifLoad = False
        # Call the network for training
        setRandom(config['randSeed'])

        net.load_state_dict(netInitState, strict=False)
        net_ = ERM_SMA(net, start=sma)

        outPathSub = os.path.join(config['outPath'], 'sub' + str(iSub))
        model = baseModel(net=net_, resultsSavePath=outPathSub, **config['baseModelArugments'])
        start = time.time()
        model.train(trainData, valData, testData, **config['modelTrainArguments'], )

        # extract the important results.
        trainResults.append([d['results']['trainBest'] for d in model.expDetails])
        valResults.append([d['results']['valBest'] for d in model.expDetails])
        testResults.append([d['results']['test'] for d in model.expDetails])
        bestEpoch.append(np.float64([d['results']['train']['bestEpoch'] for d in model.expDetails][0]))
        # save the results
        totalTime.append(np.round((time.time() - start) / 60, 2))

        results = {'train:': trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1],
                   'runTime': totalTime[-1], 'bestEpoch': bestEpoch[-1]}
        dictToCsv(os.path.join(outPathSub, 'results.csv'), results)

        save_book_simple(trainResults, valResults, testResults, subs, totalTime, bestEpoch, config,
                         name='resultsPart.xls')

    save_book(trainResults, valResults, testResults, subs, totalTime, bestEpoch, config, name='results.xls')


if __name__ == '__main__':
    # algorithms ['ce', 'coral', 'scl', 'smcldgn', 'smcldgn_mc','mixup', 'mmd', 'dann', 'irm', 'mldg']
    ho(datasetId=0, network='B7', batchSize=32, feature=32, subTorun=[0, 9], dropoutP=0., c=0.5, isProj=True,
       tradeOff=1, tradeOff2=0.1, maxEpochs=200, sma=100, algorithm='smcldgn', ps='', )

    # ho(datasetId=1, network='B71', batchSize=212, feature=32, subTorun=[0, 54], dropoutP=0, c=0.5, isProj=True,
    #    tradeOff=0.1, tradeOff2=0.01, sma=150, tradeOff4=20, maxEpochs=50, algorithm='smcldgn_mc', ps='')

    # ho(datasetId=3, network='B73', batchSize=52, feature=32, subTorun=[0, 14], dropoutP=0.3, c=2, isProj=True,
    #    tradeOff=0.001, tradeOff2=2, sma=100, maxEpochs=150, algorithm='scldgn', ps='')

    # ho(datasetId=4, network='B74', batchSize=98, feature=32, subTorun=[0, 50], dropoutP=0, c=0.5, isProj=True,
    #    tradeOff=0.1, tradeOff2=0.001, sma=10, tradeOff4=20, maxEpochs=40, algorithm='smcldgn_mc', ps='')

    # ho(datasetId=6, network='B76', batchSize=216, feature=32, subTorun=[0, 109], dropoutP=0., c=1, isProj=True,
    #    tradeOff=1, tradeOff2=0.001, sma=50, tradeOff4=20, maxEpochs=200, algorithm='smcldgn_mc', ps='')

    # ho(datasetId=7, network='B77', batchSize=216, feature=32, subTorun=[0, 1], dropoutP=0., c=0.5, isProj=True,
    #    tradeOff=1, tradeOff2=0.0005, sma=50, tradeOff4=20, maxEpochs=200, algorithm='smcldgn_mc', ps='')
