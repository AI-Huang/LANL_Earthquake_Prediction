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


def save_data(path):
    """save data into npz files
    """
    import time
    # warnings.filterwarnings("ignore")

    start = time.clock()
    # int16 for acoustic_data, and float32 for time_to_failure is enough :)
    df_train = pd.read_csv(os.path.join(path, 'data/train.csv'),
                           dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    elapsed = time.clock() - start

    print("read_csv time used:", elapsed)

    start = time.clock()

    np.savez_compressed(os.path.join(path, 'processed/train_acoustic_data.npz'),
                        acoustic_data=df_train['acoustic_data'].values)
    np.savez_compressed(os.path.join(path, 'processed/train_time_to_failure.npz'),
                        time_to_failure=df_train['time_to_failure'].values)

    elapsed = time.clock() - start
    print("savez_compressed time used:", elapsed)


def load_data(path):
    """test time during load data
    """
    start = time.clock()
    train_acoustic_data = np.load(
        os.path.join(path, 'train_acoustic_data.npz'))
    elapsed = time.clock() - start
    print("load time used:", elapsed)


def main():
    BASE_PATH = "D:\\DeepLearningData\\LANL-Earthquake-Prediction"
    save_data(path=BASE_PATH)
    load_data(path=BASE_PATH)


if __name__ == "__main__":
    main()
