import tensorflow as tf
import numpy as np
import pickle

if __name__ == "__main__":
    new_model = tf.keras.models.load_model("character_recognition.model")
    test_img = pickle.load(open("test_images", "rb"))
    test_lab = pickle.load(open("test_labels", "rb"))
    prediction = new_model.predict([test_img])
    n=0
    p=0.0
    for i in prediction:
        if np.argmax(i)==test_lab[n] :
            p+=1
        n += 1
    print(p/n)