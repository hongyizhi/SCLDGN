#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
from network.MixedConv2d import MixedConv2d

current_module = sys.modules[__name__]

debug = False


class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1, 3), stride=(1, 3))
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
                nn.LogSoftmax(dim=1))

    def calculateOutSize(self, model, nChan, nTime):
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, inputSize, nClass, dropoutP=0.25, c=0.5, isProj=False, *args, **kwargs):
        super(deepConvNet, self).__init__()

        nChan, nTime = inputSize[1], inputSize[2]
        kernalSize = (1, 10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):

        f = self.allButLastLayers(x)
        x = self.lastLayer(f)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x, f, 0


class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1), padding=(0, self.C1//2), bias=False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1), padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4),
                nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22), padding=(0, 22//2), bias=False,
                          groups=self.F1*self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), stride=(1, 1), bias=False, padding=0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8),
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
                nn.LogSoftmax(dim=1))

    def calculateOutSize(self, model, nChan, nTime):
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, inputSize, nClass, dropoutP=0.25, F1=8, D=2, C1=125, c=0.5, isProj=False, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = inputSize[2]
        self.nClass = nClass
        self.nChan = inputSize[1]
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, self.nChan, self.nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        f = self.firstBlocks(x)
        x = self.lastLayer(f)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x, f, 0


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1., **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class mixBlockRes(nn.Module):

    def __init__(self, dim, kernel_size, depthwise=False, drop_rate=0.,):
        super().__init__()

        self.MixedConv2d = MixedConv2d(in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                        stride=1, padding='', dilation=1, depthwise=depthwise, )
        self.bn = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.MixedConv2d(x)
        x = self.bn(x)
        x = shortcut + self.drop_path(x)
        return x


class B7(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B7, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 15), (1, 31), (1, 63), (1, 125)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            mixBlockRes(self.outFeature, [(1, 15), (1, 31), (1, 63), (1, 125)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(self.outFeature, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 4, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0), groups=self.outFeature),
            nn.BatchNorm2d(self.outFeature * 4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)


class B71(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B71, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 15), (1, 31), (1, 63), (1, 125)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            mixBlockRes(self.outFeature, [(1, 15), (1, 31), (1, 63), (1, 125)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(self.outFeature, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 4, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature * 4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)


class B73(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B73, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 15), (1, 31), (1, 63), (1, 125)], drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            mixBlockRes(self.outFeature, [(1, 7), (1, 15), (1, 31), (1, 63)], drop_rate=dropoutP),
            nn.Conv2d(self.outFeature, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 2, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0)),
            nn.BatchNorm2d(self.outFeature * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)


class B74(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B74, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 7), (1, 15), (1, 31), (1, 63)],  depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            mixBlockRes(self.outFeature, [(1, 7), (1, 15), (1, 31), (1, 63)],  depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(self.outFeature, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 2, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0),),
            nn.BatchNorm2d(self.outFeature * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)


class B76(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B76, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 15), (1, 31), (1, 63)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 2, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0), groups=self.outFeature),
            nn.BatchNorm2d(self.outFeature * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)


class B77(nn.Module):

    def get_size(self, inputSize):
        x = torch.ones((1, inputSize[0], inputSize[1], inputSize[2]))
        x = self.feature(x)
        f = torch.flatten(x, start_dim=1)
        return f.size()

    def __init__(self, inputSize, nClass, m, dropoutP=0, c=0.5, isProj=False, *args, **kwargs):
        super(B77, self).__init__()

        self.nBands, self.nChan, self.nTime = inputSize[0], inputSize[1], inputSize[2]
        self.nClass = nClass
        self.outFeature = m

        self.feature = nn.Sequential(
            mixBlockRes(9, [(1, 15), (1, 31), (1, 63)], depthwise=True, drop_rate=dropoutP),
            nn.Conv2d(9, self.outFeature, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(self.outFeature),

            nn.Conv2d(self.outFeature, self.outFeature * 2, kernel_size=(self.nChan, 1), stride=(1, 1),
                      padding=(0, 0),),
            nn.BatchNorm2d(self.outFeature * 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
            nn.Dropout(p=dropoutP)
        )

        size = self.get_size(inputSize)
        self.fc = LinearWithConstraint(size[1], nClass, max_norm=c, doWeightNorm=True)
        if isProj:
            self.projection = nn.Sequential(
                nn.Linear(size[1], 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        x = self.feature(x)
        feature = torch.flatten(x, start_dim=1)
        proj = self.projection(feature)
        out = self.fc(feature)
        return out, feature, F.normalize(proj, dim=1)
