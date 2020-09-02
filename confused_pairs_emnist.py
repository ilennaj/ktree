# Run Order: 1st, 2 out of 2
# Determine most confused pairs of classes in only EMNIST dataset, specifically the uppercase letters


from custompackage.load_data import *
from custompackage.load_architecture import *
from custompackage.traintestloop import *

import torch
from torch.utils.data import DataLoader
import numpy as np
import math
import torchvision
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
import os
import glob
import pandas as pd
import pickle

if not os.path.exists('results'):
    os.makedirs('results')

# Testing uppercase Letters from EMNIST only

# Initialize parameters for dataset loading
bs = 256
weighting = 'paired'
trials = 10
ds_set = ['emnist']

# Initialize for record keeping
paired_test = np.zeros((len(ds_set),trials,26,26))
for m in range(trials):
    # For each 10-class dataset
    for k, ds in enumerate(ds_set):
        # Go through each class
        for i in range(10, 36):
            t1 = i
            # and pair it with every other class
            for j in range(i+1,36):
                t2 = j

                # Load the binary classification dataloaders
                trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds)

                # Assign entirety of the datasets within each dataloader to a variable
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

#                 print(ds, m, i, j, score_test)
                
                #Record keeping
                paired_test[k,m, i-10,j-10] = score_test

                np.save('./results/confused_pairs_emnist_upper.npy', paired_test)