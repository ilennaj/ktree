## 20200513_ktrees_perm_0
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

# Select Class Set
class_set = 0

# Initialize settings
bs = 256
weighting = 'paired'
trials = 10
epochs = 5000
trees_set = [1,2,4,8,16,32]



classes = np.load('./results/20200511/classes.npy', allow_pickle=True)

if class_set == 0:
    classes = classes[0:2] # mnist fmnist
elif class_set == 1:
    classes = classes[2:4] # kmnist emnist
elif class_set == 2:
    classes = classes[4:6] # svhn usps
else:
    classes = classes[6].reshape(1,-1)


loss = np.zeros((len(classes), trials, len(trees_set)))
acc = np.zeros((len(classes), trials, len(trees_set)))

    
for j, (t1, t2, ds) in enumerate(classes):
    print(t1, t2, ds)
    trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds, permute=True)
    for i in range(trials):
        for k, trees in enumerate(trees_set):
            print(j, i, k)
            model = ktree_gen(ds=ds, Repeats=trees, Padded=True).cuda()


            
            loss_curve, acc_curve, loss[j,i,k], acc[j,i,k], model_t= train_test_ktree(model, trainloaders[i],
                                                                                  testloader, epochs = epochs, randorder=False)

    
            np.save('./results/20200513/k_tree_acc_permute_'+str(class_set)+'.npy', acc)
            np.save('./results/20200513/k_tree_loss_permute_'+str(class_set)+'.npy', loss)
