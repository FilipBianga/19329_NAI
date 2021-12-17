'''
Problem:
    The below program teaches neural network how to classify cancer cells.

    We've used UCI Breast Cancer Wisconsin (Diagnostic) Data Set.
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    CSV Version to be find in files (data.csv).

Authors:
    Adam Tomporowski, s16740
    Filip Bianga, s19329
'''

# To run this script you just have to import below modules
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.layers import Dense

# To check if TF works as expected, you can try printing its version
# Also, if training takes more time than expected, try switching to the GPU based learning
print(tf.__version__)

# Importing dataset.
# In this case I'll use csv version, it's more readable and already has classes (names)
df = pd.read_csv('data/data.csv')

# Dropping last, unwanted column
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Splitting the data into two categories. X -> independent Y -> dependent data.
x = df.drop('diagnosis', axis=1)
y = df.diagnosis

# Since it's classification problem, we've to encode target labels
lb = LabelEncoder()
y = lb.fit_transform(y)

# Splitting data to train and test parts
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Scaling the data with scikit learn StandardScaler module. It makes neural network faster.
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

# Building the model.
# tf.keras.initializers.he_uniform - draws samples from a uniform distribution
# tf.keras.input_dim - substitute of input_shape
classifier = Sequential()
classifier.add(Dense(12, kernel_initializer='he_uniform', activation='relu', input_dim=30))
classifier.add(Dense(12, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
# print(classifier.summary())

# Compiling the model.
# Optimizer - This is how the model is updated based on the data it sees and its loss function.
# Loss - This measures how accurate the model is during training. You want to minimize this function to "steer" the
# model in the right direction.
# Metrics - Used to monitor the training and testing steps. The following example uses accuracy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the AI.
model = classifier.fit(xtrain, ytrain, epochs=100)
# model = classifier.fit(xtrain, ytrain, batch_size=100, epochs=100)

# Testing accuracy score for the test data
y_pred = classifier.predict(xtest)

# Converting data
# y_pred > 0.5 -> True
# y_pred < 0.5 -> False
y_pred = (y_pred > 0.5)


# Creating confusion matrix
# TN[00]: - True Negative to wynik, w którym model poprawnie przewiduje klasę Negative.
# FP[10]: - Fałszywie dodatni to wynik, w którym model nieprawidłowo przewiduje klasę pozytywną.
# FN[01]: -False Negative to wynik, w którym model nieprawidłowo przewiduje klasę Negative.
# TP[11]: - Prawdziwie pozytywny to wynik, w którym model poprawnie przewiduje klasę pozytywną.
cm = confusion_matrix(ytest, y_pred)
score = accuracy_score(ytest, y_pred)
print('Accuracy is:', score)
plt.figure(figsize=[14, 7])
sb.heatmap(cm, annot=True)
plt.show()

# Plotting accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

# Another plot to compare it with LAB4
count_malignant = 0
count_benign = 0
for i in y_pred:
    if i:
        count_malignant += 1
    else:
        count_benign += 1

sum = count_malignant + count_benign

names = ['Malignant', 'Benign']
values = [count_malignant, count_benign]
plt.figure(figsize=(9, 9))
plt.bar(names, values)
plt.suptitle('Cancer cells diagnostics')
plt.show()
