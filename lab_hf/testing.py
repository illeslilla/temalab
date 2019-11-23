import tensorflow as tf
import pickle

if __name__ == "__main__":
    new_model = tf.keras.models.load_model("character_recognition.model")
    test_img = pickle.load(open("test_images", "rb"))
    test_lab = pickle.load(open("test_labels", "rb"))
    prediction = new_model.predict([test_img])
    print(prediction)