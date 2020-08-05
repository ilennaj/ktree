
import customrnn as cr

from custompackage.load_data import *
from custompackage.load_architecture import *
from custompackage.traintestloop import *
from custompackage.neuron_capacity import *
from custompackage.sin_ineq import *

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
from hyperopt import hp, tpe, fmin, Trials
import pickle



# Initialize settings
bs = 256
weighting = 'paired'
trials = 10

classes = np.load('./results/20200511/classes.npy', allow_pickle=True)

score_test = np.zeros((len(classes), trials))

    
for j, (t1, t2, ds) in enumerate(classes):
    print(t1, t2, ds)
    print('lda')
    trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds)
    for i in range(trials):
        print(j, i)
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

        np.save('./results/20200513/lda_score_test.npy', score_test)
