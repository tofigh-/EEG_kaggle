import numpy as np
import joblib
import cPickle as pk
import os

def load_kaggle():

    data_dir = '/Users/Tofigh/Documents/Kaggle/train/features'

    fd = open(os.path.join(data_dir, 'subj10_series1_features'))
    ch = 0
    trX = joblib.load(fd).astype(np.float)[:, :, :, ch]
    [N, nx, nx] = trX.shape
    trX = trX.reshape((N, nx, nx, 1))
    maxNorm = np.max(trX)
    meanNorm = np.mean(trX, axis=0)
    trX = (trX-meanNorm) / maxNorm
    fd = open(os.path.join(data_dir, 'subj10_series1_labels'))
    trY = np.array(pk.load(fd)).astype(np.float)

    # fd = open(os.path.join(data_dir, 'subj10_series7_features'))
    # teX = joblib.load(fd).astype(np.float)[:, :, :, ch]
    # [N, nx, nx] = teX.shape
    # teX = teX.reshape((N, nx, nx, 1))
    # maxNorm = np.max(teX)
    # meanNorm = np.mean(teX, axis=0)
    # teX = (teX - meanNorm) / maxNorm
    #
    # fd = open(os.path.join(data_dir, 'subj10_series7_labels'))
    # teY = np.array(pk.load(fd)).astype(np.float)

    trY = np.asarray(trY)
    # teY = np.asarray(teY)

    # X = np.concatenate((trX, teX), axis=0)
    # y = np.concatenate((trY, teY), axis=0)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(trX)
    np.random.seed(seed)
    np.random.shuffle(trY)

    y_vec = np.zeros((len(trY), 7), dtype=np.float)
    for i, label in enumerate(trY):
        y_vec[i, label] = 1.0

    return trX, y_vec

load_kaggle()
