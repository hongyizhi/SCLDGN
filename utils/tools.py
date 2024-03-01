import csv
import random
import numpy as np
import copy
import math
import torch
from torch.utils.data import Sampler, Subset, ConcatDataset, DataLoader
import os
import xlwt

def setRandom(seed):
    np.random.seed(seed)

    # Set torch
    torch.manual_seed(seed)
    random.seed(seed)
    # Set cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def excelAddData(worksheet, startCell, data, isNpData=False):
    if type(data) is not list:
        data = [[data]]
    elif type(data[0]) is not list:
        data = [data]
    else:
        data = data

    # write the data. starting from the given start cell.
    rowStart = startCell[0]
    colStart = startCell[1]

    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if isNpData:
                worksheet.write(rowStart + i, colStart + j, col.item())
            else:
                if colStart + j < 250:
                    worksheet.write(rowStart + i, colStart + j, col)
                else:
                    worksheet.write(rowStart + i + 6, colStart + j - 250, col)

    return worksheet


def dictToCsv(filePath, dictToWrite):
    with open(filePath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictToWrite.items():
            writer.writerow([key, value])


def save_book(trainResults, valResults, testResults, subs, totalTime, bestEpoch, config, name='results.xls'):
    trainAcc = [[r['acc'] for r in result] for result in trainResults]
    trainAcc = list(map(list, zip(*trainAcc)))
    valAcc = [[r['acc'] for r in result] for result in valResults]
    valAcc = list(map(list, zip(*valAcc)))
    testAcc = [[r['acc'] for r in result] for result in testResults]
    testAcc = list(map(list, zip(*testAcc)))

    print("Results sequence is train, val , test")
    print(trainAcc)
    print(valAcc)
    print(testAcc)

    # append the confusion matrix
    trainCm = [[r['cm'] for r in result] for result in trainResults]
    trainCm = list(map(list, zip(*trainCm)))
    trainCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in trainCm]

    valCm = [[r['cm'] for r in result] for result in valResults]
    valCm = list(map(list, zip(*valCm)))
    valCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in valCm]

    testCm = [[r['cm'] for r in result] for result in testResults]
    testCm = list(map(list, zip(*testCm)))
    testCm = [np.concatenate(tuple([cm for cm in cms]), axis=1) for cms in testCm]

    print(trainCm)
    print(valCm)
    print(testCm)
    # %% Excel writing
    book = xlwt.Workbook(encoding="utf-8")
    for i, res in enumerate(trainAcc):
        sheet1 = book.add_sheet('exp-' + str(i + 1), cell_overwrite_ok=True)
        sheet1 = excelAddData(sheet1, [0, 0], ['SubId', 'trainAcc', 'valAcc', 'testAcc', 'runTime', 'bestEpoch'])
        sheet1 = excelAddData(sheet1, [1, 0], [[sub] for sub in subs])
        sheet1 = excelAddData(sheet1, [1, 1], [[acc] for acc in trainAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 2], [[acc] for acc in valAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 3], [[acc] for acc in testAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 4], [[t] for t in totalTime], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 5], [[b] for b in bestEpoch], isNpData=True)

        # write the cm11
        for isub, sub in enumerate(subs):
            sheet1 = excelAddData(sheet1,
                                  [len(trainAcc[0]) + 5, 0 + isub * config['baseModelArugments']['classes']], sub)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 6, 0], ['train CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 7, 0], trainCm[i].tolist(), isNpData=False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 11, 0], ['val CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 12, 0], valCm[i].tolist(), isNpData=False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 17, 0], ['test CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 18, 0], testCm[i].tolist(), isNpData=False)

    book.save(os.path.join(config['outPath'], name))


def save_book_simple(trainResults, valResults, testResults, subs, totalTime, bestEpoch, config, name='results1.xls'):
    trainAcc = [[r['acc'] for r in result] for result in trainResults]
    trainAcc = list(map(list, zip(*trainAcc)))
    valAcc = [[r['acc'] for r in result] for result in valResults]
    valAcc = list(map(list, zip(*valAcc)))
    testAcc = [[r['acc'] for r in result] for result in testResults]
    testAcc = list(map(list, zip(*testAcc)))

    # %% Excel writing
    book = xlwt.Workbook(encoding="utf-8")
    for i, res in enumerate(trainAcc):
        sheet1 = book.add_sheet('exp-' + str(i + 1), cell_overwrite_ok=True)
        sheet1 = excelAddData(sheet1, [0, 0], ['SubId', 'trainAcc', 'valAcc', 'testAcc', 'runTime', 'bestEpoch'])
        sheet1 = excelAddData(sheet1, [1, 0], [[sub] for sub in subs])
        sheet1 = excelAddData(sheet1, [1, 1], [[acc] for acc in trainAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 2], [[acc] for acc in valAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 3], [[acc] for acc in testAcc[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 4], [[t] for t in totalTime], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 5], [[b] for b in bestEpoch], isNpData=True)

    if os.path.exists(os.path.join(config['outPath'], name)):
        os.remove(os.path.join(config['outPath'], name))
    book.save(os.path.join(config['outPath'], name))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def splitKfold(idx1, k, doShuffle=True):
    '''
    Split the index from given list in k random parts.
    Returns list with k sublists.
    '''
    idx = copy.deepcopy(idx1)
    lenFold = math.ceil(len(idx) / k)
    if doShuffle:
        np.random.shuffle(idx)
    return [idx[i * lenFold:i * lenFold + lenFold] for i in range(k)]


def loadSplitFold(idx, path, subNo):
    '''
    Load the CV fold details saved in json formate.
    Returns list with k sublists corresponding to the k fold splitting.
    subNo is the number of the subject to load from. starts from 0
    '''
    import json
    with open(path) as json_file:
        data = json.load(json_file)
    data = data[subNo]
    # sort the values in sublists
    folds = []
    for i in list(set(data)):
        folds.append([idx[j] for (j, val) in enumerate(data) if val == i])

    return folds


def generateBalancedFolds(idx, label, kFold=5):
    '''
    Generate a class aware splitting of the data index in given number of folds.
    Returns list with k sublists corresponding to the k fold splitting.
    '''
    from sklearn.model_selection import StratifiedKFold
    folds = []
    skf = StratifiedKFold(n_splits=kFold)
    for train, test in skf.split(idx, label):
        folds.append([idx[i] for i in list(test)])
    return folds


def get_transform(filtBank=[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32],
                            [32, 36], [36, 40]],
                  fs=250, filterType='cheby2', order=3, filtType='filter', outputType='sos'):
    # outputType: 'sos', 'ba'
    # filterType: 'filter', 'filtfilt'
    if filterType == 'butter':
        transformConfig = {'filterBankWithButter': {
            'filtBank': filtBank, 'fs': fs, 'order': order,
            'filterOutType': outputType,
            'filtType': filtType}}
    elif filterType == 'cheby2':
        # filtType = 'filter'
        transformConfig = {'filterBank': {
            'filtBank': filtBank, 'fs': fs, 'filtType': filtType}}
    elif filterType == 'fir':
        transformConfig = {'filterBankFIR': {
            'filtBank': filtBank, 'fs': fs, 'filtType': filtType}}
    else:
        transformConfig = {'filterRaw': {}}

    return transformConfig


def getModelArguments(datasetId, dropoutP=0., feature=32, c=0.5, isProj=False):
    argument = {}
    if datasetId == 0:  # bci42a
        argument = {'inputSize': (9, 22, 1000), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 4, 'isProj': isProj}
    elif datasetId == 1:  # openbmi
        argument = {'inputSize': (9, 62, 1000), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 2, 'isProj': isProj}
    elif datasetId == 2:  # bci42b
        argument = {'inputSize': (9, 3, 1000), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 2, 'isProj': isProj}
    elif datasetId == 3:  # hgd
        argument = {'inputSize': (9, 44, 1000), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 4, 'isProj': isProj}
    elif datasetId == 4:  # gist
        argument = {'inputSize': (9, 64, 750), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 2, 'isProj': isProj}
    elif datasetId == 5:  # bci3
        argument = {'inputSize': (9, 49, 300), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 2, 'isProj': isProj}
    elif datasetId == 6:  # physionet4
        argument = {'inputSize': (9, 64, 480), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 4, 'isProj': isProj}
    elif datasetId == 7:  # physionet2
        argument = {'inputSize': (9, 64, 480), 'dropoutP': dropoutP,
                    'm': feature, 'c': c, 'nClass': 2, 'isProj': isProj}
    return argument


def getBaseModelArguments(datasetId, batchSize=32, tradeOff=0., tradeOff2=0., tradeOff3=0.,
                          tradeOff4=0., algorithm='ce'):
    ndomainList = [9, 54, 9, 14, 50, 5, 109, 109]
    classesList = [4, 2, 2, 4, 2, 2, 4, 2]
    argument = {'batchSize': batchSize, 'ndomain': ndomainList[datasetId],
                'classes': classesList[datasetId], 'tradeOff': tradeOff, 'tradeOff2': tradeOff2,
                'tradeOff3': tradeOff3, 'tradeOff4': tradeOff4, 'algorithm': algorithm}
    return argument


class ConcatDatasetWithDomainLabel(ConcatDataset):
    """ConcatDataset with domain label"""

    def __init__(self, *args, **kwargs):
        super(ConcatDatasetWithDomainLabel, self).__init__(*args, **kwargs)
        self.index_to_domain_id = {}
        domain_id = 0
        start = 0
        for end in self.cumulative_sizes:
            for idx in range(start, end):
                self.index_to_domain_id[idx] = domain_id
            start = end
            domain_id += 1

    def __getitem__(self, index):
        img, target = super(ConcatDatasetWithDomainLabel, self).__getitem__(index)
        domain_id = self.index_to_domain_id[index]
        return img, target, domain_id


class RandomDomainSampler(Sampler):
    r"""Randomly sample :math:`N` domains, then randomly select :math:`K` samples in each domain to form a mini-batch of
    size :math:`N\times K`.

    Args:
        data_source (ConcatDataset): dataset that contains data from multiple domains
        batch_size (int): mini-batch size (:math:`N\times K` here)
        n_domains_per_batch (int): number of domains to select in a single mini-batch (:math:`N` here)
    """

    def __init__(self, data_source: ConcatDataset, batch_size: int, n_domains_per_batch: int):
        super(Sampler, self).__init__()
        self.n_domains_in_dataset = len(data_source.cumulative_sizes)
        self.n_domains_per_batch = n_domains_per_batch
        assert self.n_domains_in_dataset >= self.n_domains_per_batch

        self.sample_idxes_per_domain = []
        start = 0
        for end in data_source.cumulative_sizes:
            idxes = [idx for idx in range(start, end)]
            self.sample_idxes_per_domain.append(idxes)
            start = end

        assert batch_size % n_domains_per_batch == 0
        self.batch_size_per_domain = batch_size // n_domains_per_batch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        sample_idxes_per_domain = copy.deepcopy(self.sample_idxes_per_domain)
        domain_idxes = [idx for idx in range(self.n_domains_in_dataset)]
        final_idxes = []
        stop_flag = False
        while not stop_flag:
            selected_domains = domain_idxes

            for domain in selected_domains:
                sample_idxes = sample_idxes_per_domain[domain]
                if len(sample_idxes) < self.batch_size_per_domain:
                    selected_idxes = np.random.choice(sample_idxes, self.batch_size_per_domain, replace=True)
                else:
                    selected_idxes = random.sample(sample_idxes, self.batch_size_per_domain)
                final_idxes.extend(selected_idxes)

                for idx in selected_idxes:
                    if idx in sample_idxes_per_domain[domain]:
                        sample_idxes_per_domain[domain].remove(idx)

                remaining_size = len(sample_idxes_per_domain[domain])
                if remaining_size < self.batch_size_per_domain:
                    stop_flag = True

        return iter(final_idxes)

    def __len__(self):
        return self.length


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n data points in the first dataset and the rest in the last,
    using the given random seed
    """
    assert (n <= len(dataset))
    idxes = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(idxes)
    subset_1 = idxes[:n]
    subset_2 = idxes[n:]
    return Subset(dataset, subset_1), Subset(dataset, subset_2)


def split_list(lst, ratios, num_splits):
    assert (len(ratios) == num_splits)
    total_ratio = sum(ratios)
    assert (total_ratio == 1)
    n = len(lst)
    result = []
    start = 0
    for i in range(num_splits):
        end = start + int(n * ratios[i])
        result.append(lst[start:end])
        start = end
    return result


def split_idx(idx, n, seed=0, datasetId=0):
    if datasetId == 100:
        idx1 = idx[:-40]
        idx2 = idx[-40:]
        len_idx1 = len(idx1)
        len_idx2 = len(idx2)
    elif datasetId == 6 or datasetId == 7:
        ll = len(idx)
        np.random.RandomState(seed).shuffle(idx)
        t = int(ll * n) + 1
        return idx[:t], idx[t:]
    else:
        l = len(idx)
        idx1 = idx[:l//2]
        idx2 = idx[l//2:]
        len_idx1 = len(idx1)
        len_idx2 = len(idx2)
    np.random.RandomState(seed).shuffle(idx1)
    np.random.RandomState(seed).shuffle(idx2)
    n1 = int(len_idx1 * n)
    n2 = int(len_idx2 * n)
    idx_train = idx1[:n1] + idx2[:n1]
    idx_val = idx1[n2:] + idx2[n2:]
    return idx_train, idx_val


class ForeverDataIterator:

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_split(x_list, labels_list, n_domains_per_batch, n_support_domains):
    assert n_support_domains < n_domains_per_batch

    support_domain_idxes = random.sample(range(n_domains_per_batch), n_support_domains)
    support_domain_list = [(x_list[idx], labels_list[idx]) for idx in range(n_domains_per_batch) if
                           idx in support_domain_idxes]
    query_domain_list = [(x_list[idx], labels_list[idx]) for idx in range(n_domains_per_batch) if
                         idx not in support_domain_idxes]

    return support_domain_list, query_domain_list