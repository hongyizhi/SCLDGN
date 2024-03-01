#!/usr/bin/env python
# coding: utf-8

# To do deep learning
import os
import sys
import copy
import time
import pickle
import random

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data.sampler as builtInSampler
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, Path)
from utils import stopCriteria, samplers
from utils.tools import RandomDomainSampler, random_pairs_of_minibatches, random_split, ForeverDataIterator
from lossFunction import coral, scl, irm, mmd, mcc


class baseModel:
    def __init__(
            self,
            net,
            resultsSavePath=None,
            seed=961102,
            setRng=True,
            preferedDevice='gpu',
            nGPU=0,
            batchSize=1,
            tradeOff=0.,
            tradeOff2=0.,
            tradeOff3=0.,
            tradeOff4=0,
            ndomain=9,
            classes=4,
            algorithm='ce'):

        self.seed = seed
        self.preferedDevice = preferedDevice
        self.batchSize = batchSize
        self.setRng = setRng
        self.resultsSavePath = resultsSavePath
        self.device = None
        self.ndomain = ndomain - 1

        self.tradeOff = tradeOff
        self.tradeOff2 = tradeOff2
        self.tradeOff3 = tradeOff3
        self.tradeOff4 = tradeOff4
        # Set RNG
        if self.setRng:
            self.setRandom(self.seed)

        # Set device
        self.setDevice(nGPU)
        self.net = net.to(self.device)
        self.nclasses = classes
        self.classes = [i for i in range(self.nclasses)]
        self.coral = coral.CorrelationAlignmentLoss().to(self.device)
        self.scl = scl.SupConLoss().to(self.device)
        self.irm = irm.InvariancePenaltyLoss().to(self.device)
        self.mmd = mmd.MMDLoss().to(self.device)
        self.mcc = mcc.MinimumClassConfusionLoss(temperature=2.5).to(self.device)

        # ['ce', 'coral', 'scl', 'smcldgn', 'smcldgn_mc', 'irm', 'mixup', 'mmd', 'mldg']
        self.trainOneEpoch = vars(baseModel)[algorithm]

        if self.resultsSavePath is not None:
            if not os.path.exists(self.resultsSavePath):
                os.makedirs(self.resultsSavePath)
            print('Results will be saved in folder : ' + self.resultsSavePath)

    def train(
            self,
            trainData,
            valData,
            testData=None,
            optimFns='Adam',
            optimFnArgs={},
            sampler=None,
            lr=0.001,
            stopCondi=None,
            loadBestModel=True,
            bestVarToCheck='valInacc'):

        # define the classes
        classes = [i for i in range(self.nclasses)]

        # Define the sampler
        if sampler is not None:
            # sampler = self._findSampler(sampler)
            sampler = RandomDomainSampler(trainData, self.batchSize, n_domains_per_batch=self.ndomain)
        # Create the loss function
        lossFn = self._findLossFn('CrossEntropyLoss')(reduction='sum')
        # lossFn = nn.CrossEntropyLoss().cuda()
        # store the experiment details.
        self.expDetails = []

        # Lets run the experiment
        expNo = 0
        original_net_dict = copy.deepcopy(self.net.state_dict())

        # set the details
        expDetail = {'expNo': expNo, 'expParam': {'optimFn': optimFns,
                                                  'lossFn': lossFn, 'lr': lr,
                                                  'stopCondi': stopCondi}}

        # Reset the network to its initial form.
        self.net.load_state_dict(original_net_dict)

        # Run the training and get the losses.
        trainResults = self._trainOE(trainData, valData, optimFns, lr, stopCondi,
                                     optimFnArgs, classes=classes,
                                     sampler=sampler,
                                     loadBestModel=loadBestModel,
                                     bestVarToCheck=bestVarToCheck)

        # store the results and netParm
        expDetail['results'] = {'train': trainResults}
        expDetail['netParam'] = copy.deepcopy(self.net.to('cpu').state_dict())

        self.net.to(self.device)
        # If you are restoring the best model at the end of training then get the final results again.
        pred, act, l = self.predict(trainData, sampler=None)
        trainResultsBest = self.calculateResults(pred, act, classes=classes)
        trainResultsBest['loss'] = l
        pred, act, l = self.predict(valData, sampler=None)
        valResultsBest = self.calculateResults(pred, act, classes=classes)
        valResultsBest['loss'] = l
        expDetail['results']['trainBest'] = trainResultsBest
        expDetail['results']['valBest'] = valResultsBest

        # if test data is present then get the results for the test data.
        if testData is not None:
            pred, act, l = self.predict(testData, sampler=None)
            testResults = self.calculateResults(pred, act, classes=classes)
            testResults['loss'] = l
            expDetail['results']['test'] = testResults

        # Print the final output to the console:
        print("Exp No. : " + str(expNo + 1))
        print('________________________________________________')
        print("\n Train Results: ")
        print(expDetail['results']['trainBest'])
        print('\n Validation Results: ')
        print(expDetail['results']['valBest'])
        if testData is not None:
            print('\n Test Results: ')
            print(expDetail['results']['test'])

        # save the results
        if self.resultsSavePath is not None:
            # Store the graphs
            self.plotLoss(trainResults['trainLoss'], trainResults['valLoss'],
                          savePath=os.path.join(self.resultsSavePath,
                                                'exp-' + str(expNo) + '-loss.png'))
            self.plotAcc(trainResults['trainResults']['acc'],
                         trainResults['valResults']['acc'],
                         savePath=os.path.join(self.resultsSavePath,
                                               'exp-' + str(expNo) + '-acc.png'))

            # Store the data in experimental details.
            with open(os.path.join(self.resultsSavePath, 'expResults' +
                                                         str(expNo) + '.dat'), 'wb') as fp:
                pickle.dump(expDetail, fp)
            # Store the net parameters in experimental details.
            # https://zhuanlan.zhihu.com/p/129948825 store model methods.
            model_path = self.resultsSavePath + '\\checkpoint.pth.tar'
            torch.save({'state_dict': self.net.state_dict()}, model_path)

        # Increment the expNo
        self.expDetails.append(expDetail)
        expNo += 1

    def _trainOE(
            self,
            trainData,
            valData,
            optimFn='Adam',
            lr=0.001,
            stopCondi=None,
            optimFnArgs={},
            loadBestModel=True,
            bestVarToCheck='valLoss',
            classes=None,
            sampler=None, ):

        # For reporting.
        trainResults = []
        valResults = []
        testResults = []
        trainLoss = []
        valLoss = []
        loss = []
        bestNet = copy.deepcopy(self.net.state_dict())
        bestValue = float('inf')
        earlyStopReached = False
        bestEpoch = 0

        # Create optimizer
        self.optimizer = self._findOptimizer(optimFn)(self.net.parameters(), lr=lr, **optimFnArgs)
        bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())

        # Initialize the stop criteria
        stopCondition = stopCriteria.composeStopCriteria(**stopCondi)

        # lets start the training.
        monitors = {'epoch': 0, 'valLoss': 10000, 'valInacc': 1}
        doStop = False
        while not doStop:
            # train the epoch.
            start = time.time()
            loss.append(self.trainOneEpoch(self, trainData, self.optimizer, sampler=sampler))

            # evaluate the training and validation accuracy.
            pred, act, l = self.predict(trainData, sampler=None)
            trainResults.append(self.calculateResults(pred, act, classes=classes))
            trainLoss.append(l)
            monitors['trainLoss'] = l
            monitors['trainInacc'] = 1 - trainResults[-1]['acc']

            pred, act, l = self.predict(valData, sampler=None)
            valResults.append(self.calculateResults(pred, act, classes=classes))
            valLoss.append(l)
            monitors['valLoss'] = l
            monitors['valInacc'] = 1 - valResults[-1]['acc']

            runTime = time.time() - start
            # print the epoch info
            print("\t \t Epoch " + str(monitors['epoch'] + 1))
            print("Train loss = " + "%.3f" % trainLoss[-1] + " Train Acc = " +
                  "%.3f" % trainResults[-1]['acc'] +
                  ' Val Acc = ' + "%.3f" % valResults[-1]['acc'] +
                  ' Val loss = ' + "%.3f" % valLoss[-1] +
                  ' Epoch Time = ' + "%.3f" % runTime)

            if loadBestModel:
                if monitors[bestVarToCheck] < bestValue:
                    bestValue = monitors[bestVarToCheck]
                    bestNet = copy.deepcopy(self.net.state_dict())
                    bestOptimizerState = copy.deepcopy(self.optimizer.state_dict())
                    bestEpoch = monitors['epoch']
            # Check if to stop
            doStop = stopCondition(monitors)

            # Check if we want to continue the training after the first stop:
            if doStop:
                # first load the best model
                if loadBestModel and not earlyStopReached:
                    self.net.load_state_dict(bestNet)
                    self.optimizer.load_state_dict(bestOptimizerState)

                # Now check if  we should continue training:
            # update the epoch
            monitors['epoch'] += 1

        # Make individual list for components of trainResults and valResults
        t = {}
        v = {}
        for key in trainResults[0].keys():
            t[key] = [result[key] for result in trainResults]
            v[key] = [result[key] for result in valResults]

        return {'trainResults': t, 'valResults': v,
                'trainLoss': trainLoss, 'valLoss': valLoss, 'bestEpoch': bestEpoch, }

    def ce(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)
                logits, _, _ = self.net.update(x)

                loss = F.cross_entropy(logits, labels)
                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def scl(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()

                all_x = d[0].to(self.device).permute(0, 3, 1, 2)
                all_y = d[1].type(torch.LongTensor).to(self.device)
                all_d = d[2].type(torch.LongTensor).to(self.device)
                batch_size = all_y.size()[0]
                lam = np.random.uniform(0.9, 1.0)

                with torch.no_grad():
                    sorted_y, indices = torch.sort(all_y)
                    sorted_x = torch.zeros_like(all_x)
                    sorted_d = torch.zeros_like(all_d)
                    for idx, order in enumerate(indices):
                        sorted_x[idx] = all_x[order]
                        sorted_d[idx] = all_d[order]
                    intervals = []
                    ex = 0
                    for idx, val in enumerate(sorted_y):
                        if ex == val:
                            continue
                        intervals.append(idx)
                        ex = val
                    intervals.append(batch_size)

                    all_x = sorted_x
                    all_y = sorted_y
                    all_d = sorted_d

                output, _, proj = self.net.update(all_x)
                loss_ce = F.cross_entropy(output, all_y)

                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2

                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, all_y, mask=None, d=None)

                loss = loss_ce + scl_loss * self.tradeOff

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def smcldgn(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                y = d[1].type(torch.LongTensor).to(self.device)
                batch_size = y.size()[0]

                y_logit, feats, proj = self.net.update(x)

                y_logit = y_logit.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                y_coral = y.chunk(self.ndomain, dim=0)

                # coral
                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_logit[domain_i], y_coral[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i)
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.coral(f_i, f_j)

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2

                # scl
                lam = np.random.uniform(0.9, 1.0)

                sorted_y, indices = torch.sort(y)
                sorted_proj = torch.zeros_like(proj)
                for idx, order in enumerate(indices):
                    sorted_proj[idx] = proj[order]
                intervals = []
                ex = 0
                for idx, val in enumerate(sorted_y):
                    if ex == val:
                        continue
                    intervals.append(idx)
                    ex = val
                intervals.append(batch_size)

                proj = sorted_proj
                y_scl = sorted_y

                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2

                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, y_scl, mask=None)

                loss = loss_ce + self.tradeOff * loss_penalty + self.tradeOff2 * scl_loss

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def smcldgn_mc(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True, )
        for d in dataLoader:
            with torch.enable_grad():

                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                y = d[1].type(torch.LongTensor).to(self.device)
                batch_size = y.size()[0]

                y_logit, feats, proj = self.net.update(x)

                # cls
                loss_ce = F.cross_entropy(y_logit, y)

                feats = feats.chunk(self.ndomain, dim=0)

                # coral
                # select n domains
                lis = list(range(self.ndomain))
                slice = random.sample(lis, self.tradeOff4)
                loss_penalty = 0
                for domain_i in range(len(slice)):
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, len(slice)):
                        f_i = feats[slice[domain_i]]
                        f_j = feats[slice[domain_j]]
                        loss_penalty += self.coral(f_i, f_j)

                loss_penalty /= len(slice) * (len(slice) - 1) / 2

                # scl
                lam = np.random.uniform(0.9, 1.0)

                sorted_y, indices = torch.sort(y)
                sorted_proj = torch.zeros_like(proj)
                for idx, order in enumerate(indices):
                    sorted_proj[idx] = proj[order]
                intervals = []
                ex = 0
                for idx, val in enumerate(sorted_y):
                    if ex == val:
                        continue
                    intervals.append(idx)
                    ex = val
                intervals.append(batch_size)

                proj = sorted_proj
                y_scl = sorted_y

                # shuffle
                mix1 = torch.zeros_like(proj)
                mix2 = torch.zeros_like(proj)
                ex = 0
                for end in intervals:
                    shuffle_indices = torch.randperm(end - ex) + ex
                    shuffle_indices2 = torch.randperm(end - ex) + ex
                    for idx in range(end - ex):
                        mix1[idx + ex] = proj[shuffle_indices[idx]]
                        mix2[idx + ex] = proj[shuffle_indices2[idx]]
                    ex = end

                p1 = lam * proj + (1 - lam) * mix1
                p2 = lam * proj + (1 - lam) * mix2

                p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
                scl_loss = self.scl(p, y_scl, mask=None)

                loss = loss_ce + self.tradeOff * loss_penalty + self.tradeOff2 * scl_loss

                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        # return the present lass. This may be helpful to stop or continue the training.
        return running_loss.item() / len(dataLoader)

    def coral(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(x)

                y_all = y_all.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                labels = labels.chunk(self.ndomain, dim=0)

                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_all[domain_i], labels[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i)
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.coral(f_i, f_j)

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2

                loss = loss_ce + self.tradeOff * loss_penalty

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mmd(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(x)

                y_all = y_all.chunk(self.ndomain, dim=0)
                feats = feats.chunk(self.ndomain, dim=0)
                labels = labels.chunk(self.ndomain, dim=0)

                loss_ce = 0
                loss_penalty = 0
                for domain_i in range(self.ndomain):
                    # cls loss
                    y_i, labels_i = y_all[domain_i], labels[domain_i]
                    loss_ce += F.cross_entropy(y_i, labels_i)
                    # correlation alignment loss
                    for domain_j in range(domain_i + 1, self.ndomain):
                        f_i = feats[domain_i]
                        f_j = feats[domain_j]
                        loss_penalty += self.mmd(f_i, f_j)

                loss_ce /= self.ndomain
                loss_penalty /= self.ndomain * (self.ndomain - 1) / 2

                loss = loss_ce + self.tradeOff * loss_penalty

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mixup(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                data = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                minibatches = [(x.to(self.device), y.to(self.device)) for x, y in
                               zip(data.chunk(self.ndomain, dim=0), labels.chunk(self.ndomain, dim=0))]

                loss = 0
                for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
                    lam = np.random.beta(self.tradeOff, self.tradeOff)
                    # lam = np.random.uniform(self.tradeOff, 1.0)
                    x = lam * xi + (1 - lam) * xj
                    predictions, _, _ = self.net.update(x)

                    loss += lam * F.cross_entropy(predictions, yi)
                    loss += (1 - lam) * F.cross_entropy(predictions, yj)

                loss /= len(minibatches)

                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def irm(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                data = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                y_all, feats, _ = self.net.update(data)
                loss_ce = F.cross_entropy(y_all, labels)
                loss_penalty = 0
                for y_per_domain, labels_per_domain in zip(y_all.chunk(self.ndomain, dim=0),
                                                           labels.chunk(self.ndomain, dim=0)):
                    # normalize loss by domain num
                    loss_penalty += self.irm(y_per_domain, labels_per_domain) / self.ndomain

                loss = loss_ce + loss_penalty * self.tradeOff
                # backward pass
                loss.backward()
                optimizer.step()
            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def mldg(self, trainData, optimizer, sampler=None, **kwargs):

        self.net.train()
        running_loss = 0
        dataLoader = DataLoader(trainData, batch_size=self.batchSize, sampler=sampler, drop_last=True)
        n_support_domains = 3
        n_query_domains = self.ndomain - n_support_domains

        for d in dataLoader:
            with torch.enable_grad():
                # zero the parameter gradients
                # optimizer.zero_grad()
                x = d[0].to(self.device).permute(0, 3, 1, 2)
                labels = d[1].type(torch.LongTensor).to(self.device)

                x_list = x.chunk(self.ndomain, dim=0)
                labels_list = labels.chunk(self.ndomain, dim=0)
                support_domain_list, query_domain_list = random_split(x_list, labels_list, self.ndomain,
                                                                      n_support_domains)
                optimizer.zero_grad()
                with higher.innerloop_ctx(self.net, optimizer, copy_initial_weights=False) as (
                        inner_model, inner_optimizer):
                    # perform inner optimization
                    for _ in range(2):
                        loss_inner = 0
                        for (x_s, labels_s) in support_domain_list:
                            y_s, _, _ = inner_model.update(x_s)
                            # normalize loss by support domain num
                            loss_inner += F.cross_entropy(y_s, labels_s) / n_support_domains

                        inner_optimizer.step(loss_inner)

                    # calculate outer loss
                    loss_outer = 0
                    cls_acc = 0

                    # loss on support domains
                    for (x_s, labels_s) in support_domain_list:
                        y_s, _, _ = self.net.update(x_s)
                        # normalize loss by support domain num
                        loss_outer += F.cross_entropy(y_s, labels_s) / n_support_domains

                    # loss on query domains
                    for (x_q, labels_q) in query_domain_list:
                        y_q, _, _ = inner_model.update(x_q)
                        # normalize loss by query domain num
                        loss_outer += F.cross_entropy(y_q, labels_q) * self.tradeOff / n_query_domains

                loss = loss_outer
                # backward pass
                loss_outer.backward()
                optimizer.step()

            # accumulate the loss over mini-batches.
            running_loss += loss.data

        return running_loss.item() / len(dataLoader)

    def predict(self, data, sampler=None):

        predicted = []
        actual = []
        loss = 0
        self.net.eval()

        dataLoader = DataLoader(data, batch_size=self.batchSize, sampler=sampler, drop_last=False)

        # with no gradient tracking
        with torch.no_grad():
            # iterate over all the data
            for d in dataLoader:
                preds, _, _ = self.net.predict(d[0].permute(0, 3, 1, 2).to(self.device))
                loss += F.cross_entropy(preds, d[1].type(torch.LongTensor).to(self.device)).data

                # Convert the output of soft-max to class label
                _, preds = torch.max(preds, 1)
                predicted.extend(preds.data.tolist())
                actual.extend(d[1].tolist())

        return predicted, actual, loss.clone().detach().item() / len(dataLoader)

    def online(self, data):

        predicted = []
        actual = []
        loss = 0
        self.net.eval()

        dataLoader = DataLoader(data, batch_size=32, sampler=None, drop_last=False)

        # with no gradient tracking
        with torch.no_grad():
            # iterate over all the data
            for d in dataLoader:
                preds, _, _ = self.net.predict(d[0].permute(0, 3, 1, 2).to(self.device))
                loss += F.cross_entropy(preds, d[1].type(torch.LongTensor).to(self.device)).data

                # Convert the output of soft-max to class label
                _, preds = torch.max(preds, 1)
                predicted.extend(preds.data.tolist())
                actual.extend(d[1].tolist())
        r = self.calculateResults(predicted, actual, classes=self.classes)
        return r

    def calculateResults(self, yPredicted, yActual, classes=None):

        acc = accuracy_score(yActual, yPredicted)
        acc = np.round(acc, 4)
        if classes is not None:
            cm = confusion_matrix(yActual, yPredicted, labels=classes)
        else:
            cm = confusion_matrix(yActual, yPredicted)

        return {'acc': acc, 'cm': cm}

    def plotLoss(self, trainLoss, valLoss, savePath=None):

        plt.figure()
        plt.title("Training Loss vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Loss")
        plt.plot(range(1, len(trainLoss) + 1), trainLoss, label="Train loss")
        plt.plot(range(1, len(valLoss) + 1), valLoss, label="Validation Loss")
        plt.legend(loc='upper right')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            print('')
        plt.close()

    def plotAcc(self, trainAcc, valAcc, testAcc=None, savePath=None):

        plt.figure()
        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1, len(trainAcc) + 1), trainAcc, label="Train Acc")
        plt.plot(range(1, len(valAcc) + 1), valAcc, label="Validation Acc")
        if testAcc is not None:
            plt.plot(range(1, len(testAcc) + 1), testAcc, label="test Acc")
        plt.ylim((0, 1.))
        plt.legend(loc='lower left')
        if savePath is not None:
            plt.savefig(savePath)
        else:
            print('')
        plt.close()

    def setRandom(self, seed):

        self.seed = seed

        # Set np
        np.random.seed(self.seed)

        # Set torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Set cudnn
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setDevice(self, nGPU=0):

        if self.device is None:
            if self.preferedDevice == 'gpu':
                self.device = torch.device("cuda:" + str(nGPU) if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device('cpu')
            print("Code will be running on device ", self.device)

    def _findOptimizer(self, optimString):

        out = None
        if optimString in optim.__dict__.keys():
            out = optim.__dict__[optimString]
        else:
            raise AssertionError(
                'No optimizer with name :' + optimString + ' can be found in torch.optim. The list of available options in this module are as follows: ' + str(
                    optim.__dict__.keys()))
        return out

    def _findSampler(self, givenString):

        out = None
        if givenString in builtInSampler.__dict__.keys():
            out = builtInSampler.__dict__[givenString]
        elif givenString in samplers.__dict__.keys():
            out = samplers.__dict__[givenString]
        else:
            raise AssertionError('No sampler with name :' + givenString + ' can be found')
        return out

    def _findLossFn(self, lossString):

        out = None
        if lossString in nn.__dict__.keys():
            out = nn.__dict__[lossString]
        else:
            raise AssertionError(
                'No loss function with name :' + lossString + ' can be found in torch.nn. The list of available options in this module are as follows: ' + str(
                    nn.__dict__.keys()))

        return out
