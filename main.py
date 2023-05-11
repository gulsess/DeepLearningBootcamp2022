import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf

# image path
img_path = "C:/Users/musta/Downloads/spectrograms/"

folder_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
           "gun_shot", "jackhammer", "siren", "street_music"]

img_files_list = []

# creating list format image,class
for folder in folder_list:
    for img_file in os.listdir(img_path + folder):
        image = img_path + folder + "/" + img_file
        img_files_list.append([image, classes[int(folder)]])

df = pd.DataFrame(img_files_list, columns=["img", "label"])
df_labels = {
    'air_conditioner': 0,
    'car_horn': 1,
    'children_playing': 2,
    'dog_bark': 3,
    'drilling': 4,
    'engine_idling': 5,
    'gun_shot': 6,
    'jackhammer': 7,
    'siren': 8,
    'street_music': 9,

}
df['label_code'] = df['label'].map(df_labels)

X = []

dim = (32, 32)
for img in df["img"]:
    # grayscale
    img = cv2.imread(str(img))
    # resizing
    img = cv2.resize(img, dim)
    # normalization
    img = img / 255
    X.append(img)

y = df["label_code"]
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)

# Preparing model and training
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 activation="relu", ))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same",
                                 activation="relu", ))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])
X_train=tf.stack(X_train)
y_train=tf.stack(y_train)
X_val=tf.stack(X_val)
y_val=tf.stack(y_val)
results=model.fit(X_train,y_train,
                 batch_size=128,
                 epochs=50,validation_data=(X_val,y_val))
plt.plot(results.history["loss"],label="loss")
plt.plot(results.history["val_loss"],label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(results.history["accuracy"],label="accuracy")
plt.plot(results.history["val_accuracy"],label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

X_test=tf.stack(X_test)
y_test=tf.stack(y_test)
model.evaluate(X_test,y_test)
