import tensorflow as tf
import pickle



if __name__ == "__main__":
    train_img = pickle.load(open("train_images", "rb"))
    train_lab = pickle.load(open("train_labels", "rb"))

    train_img= train_img/255.0

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D((64), (3, 3), input_shape = train_img.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D((64), (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(26))
    model.add(tf.keras.layers.Activation("sigmoid"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(train_img, train_lab, batch_size=26, epochs=5, validation_split=0.1)

    model.save("character_recognition.model")

