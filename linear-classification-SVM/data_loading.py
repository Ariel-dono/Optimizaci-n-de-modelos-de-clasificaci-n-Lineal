import pickle
import numpy as np
from sklearn.datasets import load_iris

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_batch_data(path):
    data = unpickle(path)
    return [np.array(data[b'data']), np.array(data[b'labels'])]


def setting_up_img_model(img, raw_data):
    if img.size == 0:
        img = np.array(raw_data[0])
    else:
        img = np.append(img, raw_data[0], axis=0)
    return img


def load_test_data(prefixPath):
    test_data = load_batch_data(prefixPath + "/test_batch")
    return test_data


def load_raw_test_data():
    raw_data = load_test_data()
    img = np.array(raw_data[0])
    labels = np.array(raw_data[1])
    return img, labels


def setting_up_img_model(img, raw_data):
    if img.size == 0:
        img = np.array(raw_data[0])
    else:
        img = np.append(img, raw_data[0], axis=0)
    return img


def load_raw_data(prefixPath, partitions):
    img = np.array([])
    labels = np.array([])
    for i in range(0, partitions):
        raw_data = load_batch_data(prefixPath + "/data_batch_" +
                                   str(i + 1))
        img = setting_up_img_model(img, raw_data)
        labels = np.append(labels, raw_data[1], axis=0)
    return img, labels


def caching_data_pool(prefixPath, openMode, toBeCached=None):
    with open(prefixPath + "/cache", openMode) as fopened:
        if openMode == "wb":
            dict = pickle.dump(toBeCached, fopened)
        else:
            dict = pickle.load(fopened, encoding='bytes')
    return dict


def is_data_pool_cached(prefixPath):
    try:
        cache = open(prefixPath + "/cache", 'rb')
        cache.close()
        return True
    except FileNotFoundError:
        return False


def gray_scale(img):
    rowSize = int(len(img)/3)
    img.shape = (3, rowSize)
    pixelForm = []
    for i in range(0, rowSize):
        pixelForm = pixelForm + [int(np.around(0.21*int(img[0][i]) + 0.72*int(img[1][i]) + 0.07*int(img[2][i])))]
    return pixelForm


def loadCIFAR10(prefixPath):
    if is_data_pool_cached(prefixPath):
        return caching_data_pool(prefixPath, 'rb')
    else:
        imgs, labels = load_raw_data(prefixPath, 5)
        imgs4bins = []
        labels4bins = []
        for i in range(0, len(labels)):
            currentLabel = labels[i]
            if currentLabel < 4:
                currentImg = np.divide(gray_scale(imgs[i]), 255)
                currentImg = np.append(currentImg, [1], axis=0)
                imgs4bins = imgs4bins + [currentImg]
                labels4bins = labels4bins + [currentLabel]
        caching_data_pool(prefixPath, 'wb', [imgs4bins, labels4bins])
        return [imgs4bins, labels4bins]


def loadIRIS(lengthByBin, isTraining, testLengthByBin = None):
    iris = load_iris()
    data = iris.data
    labels = iris.target
    totalLength = len(data)
    patitionsSize = int(totalLength / 3)
    i = 0
    resultdata = []
    resultlabel = []
    while(len(data) > 0):
        contextData = data[:patitionsSize]
        contextLabels = labels[:patitionsSize]
        if isTraining:
            if resultdata == []:
                resultdata = np.array(contextData[:lengthByBin])
                resultlabel = np.array(contextLabels[:lengthByBin])
            else:
                resultdata = np.append(resultdata, contextData[:lengthByBin], axis=0)
                resultlabel = np.append(resultlabel, contextLabels[:lengthByBin], axis=0)
        else:
            if resultdata == []:
                resultdata = np.array(contextData[lengthByBin:])
                resultlabel = np.array(contextLabels[lengthByBin:])
            else:
                resultdata = np.append(resultdata, contextData[lengthByBin:], axis=0)
                resultlabel = np.append(resultlabel, contextLabels[lengthByBin:], axis=0)
        i += 1
        data = data[patitionsSize:]
        labels = labels[patitionsSize:]
    return [resultdata, resultlabel]
