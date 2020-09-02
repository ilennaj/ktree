# Run Order: 3rd, 2 out of 2
### Train and test lda model
### Saves test accuracy
### all classes script, early stopping implemented


from custompackage.load_data import *
from custompackage.load_architecture import *
from custompackage.traintestloop import *


import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import torchvision
from torchvision import transforms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib
import matplotlib.pyplot as plt
from torch.optim.optimizer import required
from torch.utils.data.dataset import random_split
import scipy
import os
import glob
import pandas as pd
import pickle



# Initialize settings
bs = 256
weighting = 'paired'
trials = 10

# Load class-dataset list
classes = np.load('./results/classes.npy', allow_pickle=True)

# Initialize test accuracy variable
score_test = np.zeros((len(classes), trials))

    
for j, (t1, t2, ds) in enumerate(classes):
    print(t1, t2, ds)
    print('lda')
    # Get correctly labeled and paired class datasets
    trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds)
    for i in range(trials):
        print(j, i)
        # Reassign datasets
        X_train = trainloaders[i].dataset.tensors[0]
        y_train = trainloaders[i].dataset.tensors[1]
        X_test = testloader.dataset.tensors[0]
        y_test = testloader.dataset.tensors[1]

        # initialize lda
        lda = LinearDiscriminantAnalysis()

        # fit to images, labels
        lda.fit(X_train, y_train)

        # see accuracy for validation set
        score_test[j,i] = lda.score(X_test, y_test)
        print(score_test[j,i])
        
        # Save accuracy array
        np.save('./results/lda_score_test.npy', score_test)
