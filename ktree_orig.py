# Run Order: 4th, 1 out of 3
### Train and test k-tree model
### Saves test loss and test accuracy
### Uses original image input order
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
from pytorchtools import EarlyStopping



# Test space for networks
# Select Class Set
class_set = 0

# Initialize settings
bs = 256
weighting = 'paired'
trials = 10
epochs = 2000
trees_set = [1,2,4,8,16,32]

# Load class-dataset list
classes = np.load('./results/classes.npy', allow_pickle=True)

# if class_set == 0:
#     classes = classes[0:2] # mnist fmnist
# elif class_set == 1:
#     classes = classes[2:4] # kmnist emnist
# elif class_set == 2:
#     classes = classes[4:6] # svhn usps
# else:
#     classes = classes[6].reshape(1,-1)

# Initialize final test loss and accuracy variables
loss = np.zeros((len(classes), trials, len(trees_set)))
acc = np.zeros((len(classes), trials, len(trees_set)))

# For each dataset enumerated from classes list
for j, (t1, t2, ds) in enumerate(classes):
    print(t1, t2, ds)
    # Load data loaders
    trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds, permute=False)
    # Initialize input size for model initialization purposes
    input_size = trainloaders[0].dataset.tensors[0][0].shape[0]
    # For each trial
    for i in range(trials):
        # For every k-tree defined by trees_set
        for k, trees in enumerate(trees_set):
            print(j, i, k)
            # Initialize the ktree model
            model = ktree_gen(ds=ds, Repeats=trees, Padded=True).cuda()
            
            #Train and test ktree, assigning loss and acc values
            loss_curve, acc_curve, loss[j,i,k], acc[j,i,k], model_t = train_test_ktree(model, trainloaders[i],
                                                                                  validloaders[i], testloader, epochs = epochs, randorder=False)
            # Save accuracy and loss arrays
            np.save('./results/ktree_acc_orig_'+str(class_set)+'.npy', acc)
            np.save('./results/ktree_loss_orig_'+str(class_set)+'.npy', loss)
