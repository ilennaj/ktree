import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torch.optim.optimizer import required
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer


def train_test_ktree(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    # Initialize loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    #
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            if randorder == True:
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()

            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            loss.backward()
            
####        # Freeze select weights by zeroing out gradients
            for child in model.children():
                for param in child.parameters():
                    for freeze_mask in model.freeze_mask_set:
                        if param.grad.shape == freeze_mask.shape:
                            param.grad[freeze_mask] = 0
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += (torch.round(outputs) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            if (i % 4) == 3:    # print every 80 mini-batches
                loss_curve.append(running_loss/3)
                acc_curve.append(running_acc/3)
                running_loss = 0.0
        
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output, labels.float().reshape(-1,1))
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # load the last checkpoint with the best model
#    model.load_state_dict(torch.load('checkpoint.pt'))
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            if randorder == True:
                images = images[:,ordering].cuda()
            else:
                images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()

    accuracy = correct/total
    
    print('Accuracy of the network on the test images: %2f %%' % (
        100 * accuracy))
    
    if randorder == True:
        return(loss_curve, acc_curve, loss, accuracy, model, ordering)
    else:
        return(loss_curve, acc_curve, loss, accuracy, model)

def train_test_fc(model, trainloader, validloader, testloader, epochs=10, patience=60):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    
    loss_curve = []
    acc_curve = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=False)
        
        
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += (torch.round(outputs) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            if i % 4 == 3:    # print every 80 mini-batches
                loss_curve.append(running_loss/3)
                acc_curve.append(running_acc/3)
                running_loss = 0.0
                running_acc = 0.0
            
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output, labels.float().reshape(-1,1))
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # load the last checkpoint with the best model
#    model.load_state_dict(torch.load('checkpoint.pt'))
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    correct = 0
    all_loss = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels.float().reshape(-1,1))
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
            all_loss += loss
    accuracy = correct/total
    ave_loss = all_loss.item()/total
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * accuracy))
        
    return(loss_curve, acc_curve, ave_loss, accuracy, model)