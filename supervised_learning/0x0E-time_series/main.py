#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 4 5:40:12 2021

@author: Robinson Montes
"""
import pandas as pd
import datetime as dt
preprocess = __import__('preprocess_data').preprocessing


if __name__ == "__main__":
    file_path = '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    train, validation, test = preprocess(file_path)
    print('Train values: ')
    print(train.head())
    print('Validation values:')
    print(validation.head())
    print('Test values')
    print(test.head())
