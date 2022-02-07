#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-06-25 15:49:27
# @Author  : Kelley Kan Huang (kan.huang@connect.ust.hk)
# @RefLink : https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview
# @RefLink : https://www.kaggle.com/wimwim/wavenet-lstm


import os
import time
import numpy as np
import pandas as pd
import scipy.signal as sg

# the test signals are 150k samples long, Nyquist is thus 75k.
NY_FREQ_IDX = 75000
CUTOFF = 18000
MAX_FREQ_IDX = 100000
FREQ_STEP = 2500


def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a


def fft_process(x):
    """return the FFT features
    """
    xc = x.values
    xcdm = xc - np.mean(xc)

#     b, a = des_bw_filter_lp(cutoff=18000)
#     xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcdm)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    return np.swapaxes([realFFT, imagFFT], 0, 1)


def main():
    pass


if __name__ == "__main__":
    main()
