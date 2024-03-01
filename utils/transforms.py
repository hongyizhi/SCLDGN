#!/usr/bin/env python
# coding: utf-8
"""
Helper transforms to modify the EEG data at runtime.
@author: Ravikiran Mane
"""
import copy
import numpy as np
import scipy.signal as signal
import torch


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ChangeSampleClass(object):
    """
    Change the label of the the sample.
    fromClass and toClass are the list where samples with labels in fromClass
    will be changed to have corresponding labels in toClass.
    """

    def __init__(self, fromClass, toClass):
        self.fromClass = fromClass
        self.toClass = toClass

    def __call__(self, data):
        if data['label'] in self.fromClass:
            data['label'] = self.toClass[self.fromClass.index(data['label'])]

        return data


class SelectTimeRange(object):
    """
    Select Partial time range from the trial.
    time range should be specified as a 2 element list with [nStart, nStop] (both points included)
    tAxis : the time axis.  by default its tAxis = 1 (the columns are considered time)
    """

    def __init__(self, tRange, tAxis=1):
        self.tRange = range(tRange[0], tRange[1])
        self.tAxis = tAxis

    def __call__(self, data1):
        data = copy.deepcopy(data1)
        data['data'] = np.take(data['data'], self.tRange, axis=self.tAxis)
        return data


class filterBank(object):
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance = filtAllowance
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtAllowance=2, axis=1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30  # stopband attenuation
        aPass = 3  # passband attenuation
        nFreq = fs / 2  # Nyquist frequency

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass = bandFiltCutF[1] / nFreq
            fStop = (bandFiltCutF[1] + filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            sos = signal.cheby2(N, aStop, fStop, 'lowpass', output='sos')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass = bandFiltCutF[0] / nFreq
            fStop = (bandFiltCutF[0] - filtAllowance) / nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            sos = signal.cheby2(N, aStop, fStop, 'highpass', output='sos')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass = (np.array(bandFiltCutF) / nFreq).tolist()
            fStop = [(bandFiltCutF[0] - filtAllowance) / nFreq, (bandFiltCutF[1] + filtAllowance) / nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            sos = signal.cheby2(N, aStop, fStop, 'bandpass', output='sos')

        if filtType == 'filtfilt':
            dataOut = signal.sosfiltfilt(sos, data, axis=axis)
        else:
            dataOut = signal.sosfilt(sos, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data[0]

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:, :, i] = self.bandpassFilter(d, filtBand, self.fs, self.filtAllowance,
                                               self.axis, self.filtType)
        # remove any redundant 3rd dimension
        # if len(self.filtBank) <= 1:
        #     out = np.squeeze(out, axis=2)

        data[0] = torch.from_numpy(out).float()
        return data


class filterBankFIR(object):
    """
    filter the given trial in the specific bands.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtOrder=10, axis=1, filtType='filtfilt'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtOrder = filtOrder
        self.axis = axis
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, filtOrder=50, axis=1, filtType='filter'):
        """
         Bandpass signal applying FIR filter of given order.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtOrder: order of the filter
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        # being FIR filter the a will be [1]
        a = [1]

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] == fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            h = signal.firwin(numtaps=filtOrder + 1,
                              cutoff=bandFiltCutF[1], pass_zero="lowpass", fs=fs)
        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            h = signal.firwin(numtaps=filtOrder + 1,
                              cutoff=bandFiltCutF[0], pass_zero="highpass", fs=fs)
        else:
            h = signal.firwin(numtaps=filtOrder + 1,
                              cutoff=bandFiltCutF, pass_zero="bandpass", fs=fs)

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(h, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(h, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data[0]

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])

        # check if the filter order is less than number of samples in
        # the data else set the order to nsample
        if self.filtOrder < d.shape[self.axis]:
            self.filtOrder = d.shape[self.axis]

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:, :, i] = self.bandpassFilter(d, filtBand, self.fs, self.filtOrder,
                                               self.axis, self.filtType)

        # remove any redundant 3rd dimension
        # if len(self.filtBank) <= 1:
        #     out = np.squeeze(out, axis=2)

        data[0] = torch.from_numpy(out).float()
        return data


class filterBankWithButter(object):
    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, order=3, axis=1, filterOutType='sos', filtType='filtfilt'):
        self.filtBank = filtBank
        self.fs = fs
        self.order = order
        self.axis = axis
        self.filterOutType = filterOutType
        self.filtType = filtType

    def bandpassFilter(self, data, bandFiltCutF, fs, order=3, axis=1, filterOutType='sos', filtType='filter'):

        nFreq = fs / 2  # Nyquist frequency
        wn = [bandFiltCutF[0] / nFreq, bandFiltCutF[1] / nFreq]

        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (
                bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data

        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            if filterOutType == 'sos':
                sos = signal.butter(order, wn, btype='lowpass', output=filterOutType)
            else:
                b, a = signal.butter(order, wn, 'lowpass')

        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            if filterOutType == 'sos':
                sos = signal.butter(order, wn, btype='highpass', output=filterOutType)
            else:
                b, a = signal.butter(order, wn, 'highpass')

        else:
            # band-pass filter
            # print("Using bandpass filter")
            if filterOutType == 'sos':
                sos = signal.butter(order, wn, btype='bandpass', output=filterOutType)
            else:
                b, a = signal.butter(order, wn, 'bandpass')

        if filterOutType == 'sos' and filtType == 'filtfilt':
            dataOut = signal.sosfiltfilt(sos, data, axis=axis)
        elif filterOutType == 'sos' and filtType == 'filter':
            dataOut = signal.sosfilt(sos, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)

        return dataOut

    def __call__(self, data1):

        data = copy.deepcopy(data1)
        d = data[0]

        # initialize output
        out = np.zeros([*d.shape, len(self.filtBank)])

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:, :, i] = self.bandpassFilter(d, filtBand, self.fs, self.order,
                                               self.axis, self.filterOutType, self.filtType)

        # remove any redundant 3rd dimension
        # if len(self.filtBank) <= 1:
        #     out = np.squeeze(out, axis=2)

        data[0] = torch.from_numpy(out).float()
        return data


class filterRaw(object):

    def __init__(self, ):
        self.filtBank = 0

    def __call__(self, data1):
        return data1
