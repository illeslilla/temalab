import numpy as np
import string
import os
import re
import random
import cv2
import pickle

DATADIR ="English\Img\Bmp"
CLASSES = string.ascii_uppercase
IMG_SIZE = 64


def load_filenames():
    filenames = []
    for path, dirs,files in os.walk(DATADIR):
        """Fájlrendszer bejárása és kép elérési utak betöltése"""
        filenames += list(map(lambda f: path+'\\'+f, files))
    return filenames


def create_datasets(dataset):
    test_img = []
    train_img = []

    """fájlok megkeverse és szétosztása két állományba"""
    random.shuffle(dataset)
    j=0;
    for i in dataset:
        if j<len(dataset)*0.7:
            train_img.append(i)
        else:
            test_img.append(i)
        j+=1;
    return train_img, test_img

def get_class_index(filename):
    return int(re.findall(r'.*Sample(\d+).*', filename)[0])-11


def get_class(filename):
    """Mappaszámot visszalakíom karakterré hogy megtudjam a kép milyen karaktert ábrázol"""
    return CLASSES[get_class_index(filename)]


def open_img (filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) * cv2.imread(filename.replace('Bmp', 'Msk'), cv2.IMREAD_GRAYSCALE)
    processed_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return processed_img


def get_dataset(dataset):
    data = []
    for i in dataset:
        img = open_img(i)
        lab = get_class_index(i)
        data.append([img, lab])
    return data

def save_dataset(data, img_name, lab_name):
    X = []
    Y = []

    for img, lab in data:
        X.append(img)
        Y.append(lab)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    pickle_out = open(img_name, "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open(lab_name, "wb")
    pickle.dump(Y, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    filenames = load_filenames()
    train_img, test_img = create_datasets(filenames)
    train_data = get_dataset(train_img)
    test_data = get_dataset(test_img)

    save_dataset(train_data, "train_images", "train_labels")
    save_dataset(test_data, "test_images", "test_labels")







