"""
Author: Filip Bianga

===================================
Support Vector Machines Classifier
===================================

* Code inspired by the app from the class

To run the program, install the following tools(if you dont have):
pip install numpy
pip install sklearn

python/python3 main.py

--------------------------------------------------------------------
DataSet link:

Pima Indians Diabetes
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv

Transfusions
https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center

---------------------------------------------------------------------
Output:
Pima Indians Diabetes
Output console:
1 - positice for diabetes
0 - negative fo diabetes

Blood transfusion
Output console:
1 - he/she donated blood in March 2007
0 - he/she not donated blood in March 2007
"""


import numpy as np
from pimaIndiansDiabetesClassifier import pimaIndiansDiabetesClassifier
from transfusionClassifier import transfusionClassifier


def get_indian():
    """
    Get details about Pima Indians Diabetets
    """
    print('Pima Indians Diabetes: ')
    pregnant = int(input('1. Number of times pregnant: '))
    glucose = int(input('2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test: '))
    pressure = int(input('3. Diastolic blood pressure (mm Hg): '))
    skin_thickness = int(input('4. Triceps skin fold thickness (mm): '))
    insulin = int(input('5. 2-Hour serum insulin (mu U/ml): '))
    body_mass = float(input('6. Body mass index (weight in kg/(height in m)^2): '))
    diabets = float(input('7. Diabetes pedigree function: '))
    age = int(input('8. Age (years): '))
    return np.array([pregnant, glucose, pressure, skin_thickness, insulin, body_mass, diabets, age])


def get_transfusion():
    """
    Get details about Transfusion
    """
    print("Transfusion: ")
    recency = int(input('1. Recency - months since last donation: '))
    frequency = int(input('2. Frequency - total number of donation: '))
    monetary = int(input('3. Monetary - total blood donated in c.c.: '))
    time = int(input('4. Time - months since first donation: '))
    return np.array([recency, frequency, monetary, time])


if __name__ == "__main__":

    select = int(input("Choose 1 if you want to go type Pima Indians Diabetes or choose 2 to go Transfusion: "))

    """
    We choose type
    """
    if select == 1:
        classifier = pimaIndiansDiabetesClassifier()
        data = get_indian()
    elif select == 2:
        classifier = transfusionClassifier()
        data = get_transfusion()
    else:
        print("Error")

    print("Output: ", classifier.pred(data))
