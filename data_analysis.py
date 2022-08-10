#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/6/2 15:07
"""
import numpy as np
from scipy.fftpack import fft
from compute_angles import *

# Calculate the derivative of the sequence
def cal_deriv(x, y):  # x, y is list
    diff_x = []  # dx
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)
    diff_y = []  # dy
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)
    slopes = []  # slope
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])
    deriv = []  # dy/dx
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])  # dy/dx-
    deriv.append(slopes[-1])  # dy/dx+
    # for i in deriv:  # test
    #     print(i)
    return np.array(deriv)  # return dy/dx

def fourieranalysis(x, N, dB=True):
    Fs = N  # Sampling Rate
    n = len(x)  # Sequence length
    T = n / Fs  # The number of cycles
    k = np.arange(n)  # The number of frequencies
    frq = k / T
    half_x = frq[range(int(n / 2))]
    fft_x = fft(x)
    abs_x = np.abs(fft_x)  # mod
    normalization_x = abs_x / n  # Normalized
    normalization_half_x = normalization_x[range(int(n / 2))]  # take a half
    if dB:
        normalization_half_x = 20*np.log10(normalization_half_x)
    return half_x, normalization_half_x

def compute_angles(angles_keypoints_data, rand=True):
    angles = []
    angles.append(angles_flex(angles_keypoints_data,rand))
    angles.append(angles_axis(angles_keypoints_data,rand))
    angles.append(angles_crossaxis(angles_keypoints_data,rand))
    angles = np.array(angles).T
    return angles