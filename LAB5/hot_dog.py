"""
Authors:
    Adam Tomporowski, s16740
    Filip Bianga, s19329

Hot Dog or Not Hot Dog
-----------------------------------------------------------------------
This app identifies whether something is Hot dog or not.
Well, we can train with other types of objects to identify them as well.
-----------------------------------------------------------------------
Inspiration: Series Silicon Valley
YouTube: https://www.youtube.com/watch?v=pqTntG1RXSY
Dataset: https://we.tl/t-ouym82n7IB
"""
# Based https://towardsdatascience.com/hot-dog-or-not-hot-dog-ab9d67f20674

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from PIL import Image
import glob
import random

files = glob.glob("dataset/hotdog-nothotdog/train/hotdog/*.jpg")
train_data = []
index = 0

#Setting the size of training photos hot dog
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    train_data.append((imgarray,1))

files = glob.glob("dataset/hotdog-nothotdog/train/nothotdog/*.jpg")
index = 0

#Setting the size of training photos not hot dog
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    train_data.append((imgarray,0))

random.shuffle(train_data)

files = glob.glob("dataset/hotdog-nothotdog/test/hotdog/*.jpg")
test_data = []
index = 0

#Setting the size of testing photos hot dog
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    test_data.append((imgarray,1))

files=glob.glob("dataset/hotdog-nothotdog/test/nothotdog/*.jpg")
index = 0

#Setting the size of testing photos not hot dog
for file in files:
    img = Image.open(file)
    img = img.resize((128, 128))
    imgarray = np.asarray(img)
    test_data.append((imgarray,0))

random.shuffle(test_data)


X_train = np.array([item[0] for item in train_data])
Y_train = np.array([item[1] for item in train_data])

X_test = np.array([item[0] for item in test_data])
Y_test = np.array([item[1] for item in test_data])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize to range 0-1
X_train /= 255
X_test /= 255

"""
-------------
Prepare model
-------------
"""
model = keras.Sequential()
model.add(keras.layers.AveragePooling2D((2,2),2,input_shape=(128,128,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1,activation='sigmoid'))
"""
 -----------------------
 Compile and train model
 -----------------------
"""
model.compile(optimizer ='adam',loss=keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 3, batch_size = 100)

model.evaluate(X_test,Y_test)

Predictions = model.predict(X_test)


"""
We show result from test pictures
"""
plt.figure(figsize=(10, 10))
a = 95
for i in range(0,9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i+a])
    if str(Y_test[i+a]) == '1':
        plt.title("Hot Dog")
    else:
        plt.title("Not Hot Dog")
    plt.axis("off")

plt.show()