#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 4 5:40:12 2021

@author: Robinson Montes
"""


def preprocessing(name_file):
    """
    Function to clean tha data from csv

    Arguments:
     - name_file is the name of the file that contains the data

    Returns:
     - train is the train values
     - validation is the validation values
     - test is the test values
    """

    df = pd.read_csv(name_file)
    df = df.dropna()
    df['Timestamp'] = pd.to_datetime(db_data['Timestamp'], unit='s')
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(inplace=True, drop=True)
    df = df.drop(['Timestamp'], axis=1)
    df = df[0::60]

    n = len(df)

    # Split data
    X_train = df[0:int(n * 0.7)]
    X_val = df[int(n * 0.7):int(n * 0.9)]
    X_test = df[int(n * 0.9):]

    # Normalize data
    X_mean = train.mean()
    X_std = train.std()
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    return X_train, X_val, X_test
