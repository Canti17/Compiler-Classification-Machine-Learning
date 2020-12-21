import random
import numpy as np
import pandas as pd
import jsonlines as js
import csv
import sklearn.feature_extraction as skf
from sklearn.model_selection import train_test_split 
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import *
from sklearn.tree import *
import sklearn.metrics
from sklearn.metrics import *
import scikitplot as skplt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def exerciseBinary(dataset_train, test_blind):
    vectoresempio = skf.text.CountVectorizer()
    vectorTrainBlind = vectoresempio.fit_transform(dataset_train['instructions'])
    vectorBlind = vectoresempio.transform(test_blind)
    clf = DecisionTreeClassifier(random_state = 0).fit(vectorTrainBlind,dataset_train['opt'])
    y_predDecisionTree = clf.predict(vectorBlind)
    print(y_predDecisionTree)
    return y_predDecisionTree


def exerciseMulticlass(dataset_train, test_blind):
    vectoresempio = skf.text.CountVectorizer()
    vectorTrainBlind = vectoresempio.fit_transform(dataset_train['instructions'])
    vectorBlind = vectoresempio.transform(test_blind)
    #vector1 = vectorTrainBlind.toarray()
    #vector2 = vectorBlind.toarray()
    clf = DecisionTreeClassifier(random_state=0).fit(vectorTrainBlind,dataset_train['compiler'])
    y_predMultinomial = clf.predict(vectorBlind)
    print(y_predMultinomial)
    return y_predMultinomial


                
def createthecsv(input_file,file_csv,targetBinario,targetMulticlasse):
    with js.open(input_file) as file:
        with open(file_csv,mode='w+') as csv_f:

            writer = csv.writer(csv_f)
            writer.writerow(('compiler','opt'))
            for i in range(3000):
                writer.writerow((targetMulticlasse[i],targetBinario[i]))




    
    
