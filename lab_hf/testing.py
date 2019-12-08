import tensorflow as tf
import numpy as np
import pickle
import string
import matplotlib.pyplot as plt

CLASSES = string.ascii_uppercase


def get_class(index):
    """Mappaszámot visszalakíom karakterré hogy megtudjam a kép milyen karaktert ábrázol"""
    return CLASSES[index]


if __name__ == "__main__":
    new_model = tf.keras.models.load_model("character_recognition.model")
    test_img = pickle.load(open("test_images", "rb"))
    test_lab = pickle.load(open("test_labels", "rb"))
    prediction = new_model.predict([test_img])
    tomb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tomb2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n=0
    for i in prediction:
        tomb[np.argmax(i)-1]+=1;
        if np.argmax(i)==test_lab[n]:
            tomb2[np.argmax(i)-1]+=1;
        n += 1
    tomb3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    n=0
    for i in tomb:
        tomb3[n]=tomb2[n]/i
        print(tomb3[n])
        n+=1
