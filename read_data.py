import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_ml1m(filepath, sep='::', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    dataf = pd.read_csv(filepath, sep=sep, header=None,engine='python').iloc[:, :3]-1
    df = dataf.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict, dataf[0].max()+1, dataf[1].max()+1


def read_lastfm(filepath, sep='\t', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    dataf = pd.read_csv(filepath, sep=sep, header=header,engine='python').iloc[:, :3]
    dataf['userID'] = pd.factorize(dataf['userID'])[0].astype(int)
    dataf['artistID'] = pd.factorize(dataf['artistID'])[0].astype(int)
    df = dataf.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict, dataf['userID'].max()+1, dataf['artistID'].max()+1


def read_steam(filepath, sep=',', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    dataf = pd.read_csv(filepath, sep=sep, header=None,engine='python').iloc[:, :3]
    purchased = dataf.groupby(2).get_group('purchase').copy()
    purchased[0] = pd.factorize(purchased[0])[0].astype(int)
    purchased[1] = pd.factorize(purchased[1])[0].astype(int)
    df = purchased.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict, purchased[0].max()+1, purchased[1].max()+1


def read_amazon(filepath, sep=',', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    dataf = pd.read_csv(filepath, sep=sep, header=header,engine='python').iloc[:, :3]
    dataf['reviewerID'] = pd.factorize(dataf['reviewerID'])[0].astype(int)
    dataf['asin'] = pd.factorize(dataf['asin'])[0].astype(int)
    df = dataf.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict, dataf['reviewerID'].max()+1, dataf['asin'].max()+1


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = 1
    return train_set, test_set