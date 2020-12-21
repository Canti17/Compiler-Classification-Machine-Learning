from operator import itemgetter
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
from Exercise import *
print("Libraries imported.")
typebm = 0

#FUNZIONE CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
   
    
    # Only use the labels that appear in the data
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#READING JSONL FILE
datasetone = pd.read_json('train_dataset.jsonl', lines=True)
datasetone['instructions'] = datasetone['instructions'].apply(','.join)

datasettwo = pd.read_json('test_dataset_blind.jsonl', lines= True)
datasettwo['instructions'] = datasettwo['instructions'].apply(','.join)

#CREATE TARGETS
targetBinary=datasetone['opt'].values
targetMulticlass=datasetone['compiler'].values

'''
#CREATE VECTORS
vectorizer = skf.text.CountVectorizer()

vector = vectorizer.fit_transform(datasetone['instructions'])

print(vector)



#CHOICES ON VECTORIZER AND SPLIT THE DATA IN TRAIN/TEST DATASET
typebm = input("Binary or MultiClass ? (1 for Binary and 2 for Multiclass): ")
X_all = vector
if typebm == "1":
    y_all = targetBinary
else:
    y_all = targetMulticlass



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=15)


#CREATE MODELS

modelBernoulli = BernoulliNB().fit(X_train, y_train)
modelMultinomial = MultinomialNB().fit(X_train, y_train)
modelDecisionTree = DecisionTreeClassifier(random_state=15).fit(X_train, y_train)


#TESTING THE MODELS
y_predBernoulli = modelBernoulli.predict(X_test)
y_predMultinomial = modelMultinomial.predict(X_test)
y_predDecisionTree = modelDecisionTree.predict(X_test)

print("\n")
print("\n")
print("BERNOULLI")
print(classification_report(y_test, y_predBernoulli))
print("-Accuracy: ", accuracy_score(y_test, y_predBernoulli))
print("-Recall: ", recall_score(y_test, y_predBernoulli, average=  None))
print("-Precision: ", precision_score(y_test, y_predBernoulli, average=  None))

print("\n")
print("\n")
print("\n")
print("MULTINOMIAL")
print(classification_report(y_test, y_predMultinomial))
print("-Accuracy: ", accuracy_score(y_test, y_predMultinomial))
print("-Recall: ", recall_score(y_test, y_predMultinomial, average=None))
print("-Precision: ", precision_score(y_test, y_predMultinomial, average=None))

print("\n")
print("\n")
print("\n")

print("DECISION TREE")
print(classification_report(y_test, y_predDecisionTree))
print("-Accuracy: ", accuracy_score(y_test, y_predDecisionTree))
print("-Recall: ", recall_score(y_test, y_predDecisionTree, average=  None))#print("-Precision: ", precision_score(y_test, y_predDecisionTree, average=  None))
print("\n")
print("\n")
print("\n")
print("\n")



if typebm == "1":
    names = ["H","L"]
    
    #predicted_probas1 = modelBernoulli.predict_proba(X_test)
    #predicted_probas2 = modelMultinomial.predict_proba(X_test)
    predicted_probas3 = modelDecisionTree.predict_proba(X_test)

    #a=confusion_matrix(y_test, y_predBernoulli,labels=None, sample_weight=None)
    #plot_confusion_matrix(y_test, y_predMultinomial,classes = names, normalize=False)
    #b=confusion_matrix(y_test, y_predMultinomial,labels=None, sample_weight=None)
    #plot_confusion_matrix(y_test, y_predMultinomial,classes = names, normalize=False)
    c=confusion_matrix(y_test, y_predDecisionTree,labels=None, sample_weight=None)
    plot_confusion_matrix(y_test, y_predDecisionTree,classes = names, normalize=False)  

    #skplt.metrics.plot_roc(y_test, predicted_probas1)
    #skplt.metrics.plot_roc(y_test, predicted_probas2)
    skplt.metrics.plot_roc(y_test, predicted_probas3)
   
    
else:
    names=["gcc","clang","icc"]

    #predicted_probas1 = modelBernoulli.predict_proba(X_test)
    #predicted_probas2 = modelMultinomial.predict_proba(X_test)
    predicted_probas3 = modelDecisionTree.predict_proba(X_test)

    #a=confusion_matrix(y_test, y_predBernoulli,labels=None, sample_weight=None)
    #plot_confusion_matrix(y_test, y_predBernoulli, classes = names, normalize=False)
    #b=confusion_matrix(y_test, y_predMultinomial,labels=None, sample_weight=None)
    #plot_confusion_matrix(y_test, y_predMultinomial, classes = names, normalize=False)
    c=confusion_matrix(y_test, y_predDecisionTree,labels=None, sample_weight=None)
    plot_confusion_matrix(y_test, y_predDecisionTree, classes = names, normalize=False)

    #skplt.metrics.plot_roc(y_test, predicted_probas1)
    #skplt.metrics.plot_roc(y_test, predicted_probas2)
    skplt.metrics.plot_roc(y_test, predicted_probas3)


plt.show()  

'''
ypredBinary = exerciseBinary(datasetone, datasettwo["instructions"])
ypredMulticlass = exerciseMulticlass(datasetone, datasettwo["instructions"])

count_gcc = 0
count_icc = 0
count_clang = 0
for i in ypredMulticlass[:1000]: 
  if(i=="gcc"): count_gcc += 1
for i in ypredMulticlass[1000:2000]: 
  if(i=="icc"): count_icc += 1
for i in ypredMulticlass[2000:3000]: 
  if(i=="clang"): count_clang += 1
print(count_gcc)
print(count_icc)
print(count_clang)

createthecsv('test_dataset_blind.jsonl','1707633.csv',ypredBinary,ypredMulticlass)




