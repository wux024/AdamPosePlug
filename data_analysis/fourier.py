#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2022/8/21 16:28
"""
import numpy as np
from scipy.fftpack import fft

def fourier_analysis(x, N, dB=True, retype = False):
    Fs = N            # Sampling Rate
    n = len(x)        # Sequence length
    T = n / Fs        # The number of cycles
    k = np.arange(n)  # The number of frequencies
    frq = k / T
    half_x = frq[range(int(n / 2))]
    fft_x = fft(x)
    abs_x = np.abs(fft_x)  # mod
    normalization_x = abs_x / n  # Normalized
    normalization_half_x = normalization_x[range(int(n / 2))]  # take a half
    if dB:
        normalization_half_x = 20*np.log10(normalization_half_x)
    if retype:
        return half_x, normalization_half_x
    else:
        return normalization_half_x