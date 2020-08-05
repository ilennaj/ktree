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



# Uppercase Letters

bs = 256
weighting = 'paired'
trials = 10
ds_set = ['emnist']


paired_test = np.zeros((len(ds_set),trials,26,26))
for m in range(trials):
    for k, ds in enumerate(ds_set):
        for i in range(10, 36):
            t1 = i
            for j in range(i+1,36):
                t2 = j

                trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds)

                X_train = trainloaders[0].dataset.tensors[0]
                y_train = trainloaders[0].dataset.tensors[1]
                X_test = testloader.dataset.tensors[0]
                y_test = testloader.dataset.tensors[1]


                # initialize lda
                lda = LinearDiscriminantAnalysis()

                # fit to images, labels
                lda.fit(X_train, y_train)

                # see accuracy for validation set
                score_test = lda.score(X_test, y_test)

                print(ds, m, i, j, score_test)

                paired_test[k,m, i-10,j-10] = score_test

                np.save('./results/20200509/confused_pairs_emnist_upper.npy', paired_test)