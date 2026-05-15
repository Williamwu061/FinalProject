from sklearn.decomposition import PCA
import numpy as np
import random
import scipy.io as sio
import os
import math

def load(dataset):
    data_path = os.path.join(os.getcwd(),'data')
    if dataset == 'Samson':
        X = sio.loadmat(os.path.join(data_path, 'Samson.mat'))['V']
        Y = sio.loadmat(os.path.join(data_path, 'Samson_end3.mat'))['A']
    elif dataset == 'Jasper':
        X = sio.loadmat(os.path.join(data_path, 'jasperRidge2_R198.mat'))['Y']
        Y = sio.loadmat(os.path.join(data_path, 'Japser_end4.mat'))['A']
    elif dataset == 'Urban':
        X = sio.loadmat(os.path.join(data_path, 'Urban_R162.mat'))['Y']
        Y = sio.loadmat(os.path.join(data_path, 'Urban_end4_groundTruth.mat'))['A']
    else:
        raise NotImplementedError

    return X, Y

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1],y.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex, :] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, test_ratio):
    ran = math.ceil(test_ratio*y.shape[0])
    all_ran=random.sample(range(0,y.shape[0]),y.shape[0])
    train_ran=random.sample(range(0,y.shape[0]),ran)
    test_ran = list(set(all_ran) - set(train_ran))
    Xtrain = X[train_ran,:,:,:]
    ytrain = y[train_ran,:]
    Xtest = X[test_ran,:,:,:]
    ytest = y[test_ran,:]
    return Xtrain, ytrain, Xtest, ytest
