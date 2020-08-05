
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
epochs = 500

classes = np.load('./results/20200511/classes.npy', allow_pickle=True)

loss = np.zeros((len(classes), trials))
acc = np.zeros((len(classes), trials))

    
for j, (t1, t2, ds) in enumerate(classes):
    print(t1, t2, ds)
    trainloaders, validloaders, testloader = dataset_weighted_split_all(bs, t1, t2, weighting, trials, ds)
    input_size = trainloaders[0].dataset.tensors[0][0].shape[0]
    for i in range(trials):
        print(j, i)
        model = simple_fcnn(input_size, input_size, 1).cuda()

        loss_curve, acc_curve, loss[j,i], acc[j,i], model_t = train_test_fc(model, trainloaders[i],
                                              validloaders[i], epochs = epochs)

        np.save('./results/20200513/fc_acc.npy', acc)
        np.save('./results/20200513/fc_loss.npy', loss)


