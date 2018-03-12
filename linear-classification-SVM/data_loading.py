import pickle
import numpy as np
import cv2


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


def load_test_data():
    test_data = load_batch_data("/home/ariel/Documents/Inteligencia Artificial/cifar-10-python/test_batch")
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


def load_raw_data(partitions):
    img = np.array([])
    labels = np.array([])
    for i in range(0, partitions):
        raw_data = load_batch_data("/home/ariel/Documents/Inteligencia Artificial/cifar-10-python/data_batch_" +
                                   str(i + 1))
        img = setting_up_img_model(img, raw_data)
        labels = np.append(labels, raw_data[1], axis=0)
    return img, labels

def caching_data_pool(openMode, toBeCached=None):
    with open('/home/ariel/Documents/Inteligencia Artificial/cifar-10-python/cache', openMode) as fopened:
        if openMode == "wb":
            dict = pickle.dump(toBeCached, fopened)
        else:
            dict = pickle.load(fopened, encoding='bytes')
    return dict


def is_data_pool_cached():
    try:
        cache = open("/home/ariel/Documents/Inteligencia Artificial/cifar-10-python/cache", 'rb')
        cache.close()
        return True
    except FileNotFoundError:
        return False


def gray_scale(img):
    rowSize = int(len(img)/3)
    img.shape = (3, rowSize)
    pixelForm = []
    for i in range(0, rowSize):
        pixelForm = pixelForm + [np.around(np.divide(int(img[0][i]) + int(img[1][i]) + int(img[2][i]), 3))]
    return pixelForm


def loadCIFAR10():
    if is_data_pool_cached():
        return caching_data_pool('rb')
    else:
        imgs, labels = load_raw_test_data()
        imgs4bins = []
        labels4bins = []
        for i in range(0, len(labels)):
            currentLabel = labels[i]
            if currentLabel < 4:
                currentImg = np.divide(gray_scale(imgs[i]), 255)
                currentImg = np.append(currentImg, [1], axis=0)
                imgs4bins = imgs4bins + [currentImg]
                labels4bins = labels4bins + [currentLabel]
        caching_data_pool('wb', [imgs4bins, labels4bins])
        return [imgs4bins, labels4bins]
