'''
Problem:
    The below program teaches neural network how to classify images of clothing.

    We've used Keras built-in dataset. You can find more info here:
    https://keras.io/api/datasets/fashion_mnist/

Authors:
    Adam Tomporowski, s16740
    Filip Bianga, s19329
'''

# To run this script you just have to import below modules
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# To check if TF works as expected, you can try printing its version
# Also, if training takes more time than expected, try switching to the GPU based learning
# print(tf.__version__)

# Importing and loading the data.
# Dataset uses, by default, 60k images for training and 10k images for testing.
clothes = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = clothes.load_data()

# Since clothes names are not included in the dataset, we have to assign them manually.
clothes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                 'Ankle boot']

# Before providing the data to a neural network, it must be divided by 255.
# Each pixels stores data in range from 0 to 255.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the model.
# tf.keras.layers.Flatten - transforms two-dimensional array to a one-dimensional array
# tf.keras.layers.Dense - first layer (of 128) represents neurons. The second one stands for category of clothing, we
# have 10 of them.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compiling the model.
# Optimizer - This is how the model is updated based on the data it sees and its loss function.
# Loss - This measures how accurate the model is during training. You want to minimize this function to "steer" the
# model in the right direction.
# Metrics - Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the
# images that are correctly classified.
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training the AI.
model.fit(train_images, train_labels, epochs=15)

# Using our model to predict clothes category.
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# The vector of raw (non-normalized) predictions that a classification model generates, which is ordinarily then passed
# to a normalization function. If the model is solving a multi-class classification problem, logits typically become an
# input to the softmax function. The softmax function then generates a vector of (normalized) probabilities with one
# value for each possible class.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    '''
    Plot the first X test images, their predicted labels, and the true labels.
    Color correct predictions in blue and incorrect predictions in red.

    :param i: Specifics order in data chain
    :param predictions_array: Data predicted by our model
    :param true_label: Label of provided data
    :param img: Image of provided data
    :return: Image plot
    '''
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(clothes_names[predicted_label], 100 * np.max(predictions_array),
                                         clothes_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    '''
    Plot the first X test images, their predicted labels, and the true labels.
    Color correct predictions in blue and incorrect predictions in red.

    :param i: Specifics order in data chain
    :param predictions_array: Value predicted by our model
    :param true_label: Label of provided data
    :return: Plot of predicted value
    '''
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i + 100, predictions[i + 100], test_labels)
plt.tight_layout()
plt.show()
