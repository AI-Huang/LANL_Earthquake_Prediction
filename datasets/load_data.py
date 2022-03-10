#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-06-25 15:49:27
# @Author  : Kelley Kan Huang (kan.huang@connect.ust.hk)
# @RefLink : https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview
# @RefLink : https://www.kaggle.com/wimwim/wavenet-lstm


import os
import time
from unicodedata import name
import numpy as np
import pandas as pd


def preprocess_data(path):
    """Preprocess and save data into npz files
    """
    import time
    # warnings.filterwarnings("ignore")

    start = time.clock()
    # int16 for acoustic_data, and float32 for time_to_failure is enough :)
    df_train = pd.read_csv(os.path.join(path, 'data/train.csv'),  # 9 GB
                           dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    elapsed = time.clock() - start

    # 16 GB RAM
    # CPU AMD 5800X
    # read_csv time used: 74.23s

    print("read_csv time used:", elapsed)

    start = time.clock()

    np.savez_compressed(os.path.join(path, 'processed/train_acoustic_data.npz'),
                        acoustic_data=df_train['acoustic_data'].values)
    np.savez_compressed(os.path.join(path, 'processed/train_time_to_failure.npz'),
                        time_to_failure=df_train['time_to_failure'].values)

    elapsed = time.clock() - start
    print("savez_compressed time used:", elapsed)
    # savez_compressed time used: 119.43s


def load_data(path, name):
    """test time during load data
    """
    assert name in ["train_acoustic_data", "train_time_to_failure"]
    start = time.clock()
    data = np.load(
        os.path.join(path, f"processed/{name}.npz"))
    elapsed = time.clock() - start
    print(f"load time used: {elapsed} seconds.")
    return data[name.replace("train_", "")]


def main():
    BASE_PATH = "E:\\DeepLearningData\\LANL-Earthquake-Prediction"
    preprocess_data(path=BASE_PATH)
    return
    train_acoustic_data = load_data(path=BASE_PATH, name="train_acoustic_data")
    print(train_acoustic_data.shape)

    train_time_to_failure = load_data(
        path=BASE_PATH, name="train_time_to_failure")
    print(train_time_to_failure.shape)


if __name__ == "__main__":
    main()
