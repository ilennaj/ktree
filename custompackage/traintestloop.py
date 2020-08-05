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

# Modified SGD optimizer for use with simplemodel - updates gradients sparsely
class SGD_sparse(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_sparse, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_sparse, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, mask_1, mask_2, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
#                 print('printing from within sgd',p.grad.data)
                if d_p.size() == mask_1.size():
                    d_p[mask_1] = 0
                if d_p.size() == mask_2.size():
                    d_p[mask_2] = 0
                p.data.add_(-group['lr'], d_p)
                
                
class SGD_modified(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    
    modified step method to take freezing mask

    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_modified, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_sparse, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, freeze_mask, soma_mask, closure=None):
        """Performs a single optimization step.

        Arguments:
            mask: freeze mask from model
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
#                 print('printing from within sgd',p.grad.data)
                if d_p.size() == freeze_mask.size():
                    d_p[freeze_mask] = 0
                if d_p.size() == soma_mask.size():
                    d_p[soma_mask] = 0
                p.data.add_(-group['lr'], d_p)
                
class Adam_modified(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_modified, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, freeze_mask, soma_mask, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                # Modification - apply freeze mask
                if exp_avg.size() == freeze_mask.size():
                    exp_avg[freeze_mask] = 0
                if exp_avg.size() == soma_mask.size():
                    exp_avg[soma_mask] = 0
                    
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
    
# Testing NN model(s)

# Defining training function, outputting loss and accuracy readouts

def train(Train_loader, Model, Criterion, Optimizer, Seq_length, Epochs):
    # Start timer for feedback purposes
    start = time.time()
    print('Time Since Start: %s' % (timeSince(start)))
    
    Loss_readout, Acc_readout = [],[]
    loss_current, accuracy_current = 0,0
    bin_size = 10
    
    for i in range(Epochs):
        for j, (data, labels_b, labels_d) in enumerate(Train_loader):
            # initialize inputs and target labels
            inputs, target = data.cuda(), labels_b.cuda()
            # initialize context state as an input to network
            c_state = Model.init_c_state().cuda()
            # Zero the gradients of the network
            Model.zero_grad()
            # Run model to generate outputs and new c_state.
            # Run model for as many times it takes to complete a sequence
            input_size = Model.input_size
            for k in range(Seq_length):
                inputs_seq = inputs[:,input_size*k:input_size*(k+1)]
                outputs, c_state = Model(inputs_seq, c_state)
            # Line needed when doing BCELoss in binary classification
            target_f = target.view(target.size()[0],1).float() 
            #Determine loss and optimize network
            loss = Criterion(outputs, target_f)
            loss.backward()
            Optimizer.step()
            
            loss_current += loss.item()
            
            outputs_pred = torch.zeros(inputs.size()[0]).long().cuda()
            outputs_pred[outputs.view(inputs.size()[0])>0.5] = 1
            
            accuracy_current += (outputs_pred==target).sum().item()/inputs.size()[0]
            
            if j % bin_size == 0:
                Loss_readout.append(loss_current/bin_size)
                loss_current = 0
                Acc_readout.append(accuracy_current/bin_size)
                accuracy_current = 0
    # Start timer for feedback purposes
    print('Time Since Start: %s' % (timeSince(start)))
    
    return Loss_readout, Acc_readout, outputs, outputs_pred, target

def train_simple(Train_loader, Model, Criterion, Optimizer, Seq_length, Epochs):
    # Start timer for feedback purposes
    start = time.time()
    print('Time Since Start: %s' % (timeSince(start)))
    
    Loss_readout, Acc_readout = [],[]
    loss_current, accuracy_current = 0,0
    bin_size = 10
    
    for i in range(Epochs):
        for j, (data, labels_b, labels_d) in enumerate(Train_loader):
            # initialize inputs and target labels
            inputs, target = data.cuda(), labels_b.cuda()
            # initialize context state as an input to network
            recurrent = Model.init_recurrent().cuda()
            # Zero the gradients of the network
            Model.zero_grad()
            # Run model to generate outputs and new c_state.
            # Run model for as many times it takes to complete a sequence
            input_size = Model.input_size
            #for k in range(timesteps):
            #    outputs, recurrent = Model(inputs, recurrent)
            for k in range(Seq_length):
                inputs_seq = inputs[:,input_size*k:input_size*(k+1)]
                outputs, recurrent = Model(inputs_seq, recurrent)
            # Line needed when doing BCELoss in binary classification
            target_f = target.view(target.size()[0],1).float() 
            #Determine loss and optimize network
            loss = Criterion(outputs, target_f)
            loss.backward()
            Optimizer.step(Model.mask_1, Model.mask_2) #Adding mask to force gradients to not change
            
            loss_current += loss.item()
            
            outputs_pred = torch.zeros(Batch_size).long().cuda()
            outputs_pred[outputs.view(Batch_size)>0.5] = 1
            
            accuracy_current += (outputs_pred==target).sum().item()/Batch_size
            
            if j % bin_size == 0:
                Loss_readout.append(loss_current/bin_size)
                loss_current = 0
                Acc_readout.append(accuracy_current/bin_size)
                accuracy_current = 0
    # Start timer for feedback purposes
    print('Time Since Start: %s' % (timeSince(start)))
    
    return Loss_readout, Acc_readout, outputs, outputs_pred, target

def test(Test_loader, Model, Seq_length):
    start = time.time()
    print('Time Since Start: %s' % (timeSince(start)))
    
    Batch_size = Test_loader.batch_size
    Set_size = Test_loader.dataset.tensors[1].size()[0]
    
    current_correct, correct_positives, correct_negatives = 0,0,0
    positives, negatives, total = 1, 1, 1
    all_labels_b =torch.LongTensor().cuda()
    all_labels_d =torch.LongTensor().cuda()
    all_outputs_pred = torch.LongTensor().cuda()
    
    for j, (data, labels_b, labels_d) in enumerate(Test_loader):
        # initialize inputs and target labels
        inputs, target = data.cuda(), labels_b.cuda()
        # initialize context state as an input to network
        c_state = Model.init_c_state().cuda()
        # Run model to generate outputs and new c_state.
        # Run model for as many times it takes to complete a sequence
        input_size = Model.input_size

        for k in range(Seq_length):
            inputs_seq = inputs[:,input_size*k:input_size*(k+1)]
            outputs, c_state = Model(inputs_seq, c_state)

        outputs_pred = torch.zeros(Batch_size).long().cuda()
        outputs_pred[outputs.view(Batch_size)>0.5] = 1
        
        all_labels_b = torch.cat((all_labels_b,labels_b.cuda()),0)
        all_labels_d = torch.cat((all_labels_d,labels_d.cuda()),0)

        all_outputs_pred = torch.cat((all_outputs_pred,outputs_pred),0)

        current_correct += (outputs_pred == target).sum().item()
        correct_positives += ((outputs_pred == target) & (target == 1)).sum().item()
        correct_negatives += ((outputs_pred == target) & (target == 0)).sum().item()
        positives += (target == 1).sum().item()
        negatives += (target == 0).sum().item()
        total += target.size()[0]
        
    accuracy = current_correct/total
    pos_accuracy = correct_positives/positives
    neg_accuracy = correct_negatives/negatives 
        
#     print('Time Since Start: %s' % (timeSince(start)))
    return accuracy, pos_accuracy, neg_accuracy, all_labels_b, all_labels_d, all_outputs_pred

def test_simple(Test_loader, Model):
    start = time.time()
    print('Time Since Start: %s' % (timeSince(start)))
    
    Batch_size = Test_loader.batch_size
    Set_size = Test_loader.dataset.tensors[1].size()[0]
    
    current_correct, correct_positives, correct_negatives = 0,0,0
    positives, negatives, total = 1, 1, 1
    all_labels_b =torch.LongTensor().cuda()
    all_labels_d =torch.LongTensor().cuda()
    all_outputs_pred = torch.LongTensor().cuda()
    
    for j, (data, labels_b, labels_d) in enumerate(Test_loader):
        # initialize inputs and target labels
        inputs, target = data.cuda(), labels_b.cuda()
        # initialize recurrent valjues as an input to network
        recurrent = Model.init_recurrent().cuda()
        # Run model to generate outputs and new c_state.
        # Run model for as many times it takes to complete a sequence
        input_size = Model.input_size

        for k in range(seq_length):
            inputs_seq = inputs[:,input_size*k:input_size*(k+1)]
            outputs, recurrent = Model(inputs_seq, recurrent)

        outputs_pred = torch.zeros(Batch_size).long().cuda()
        outputs_pred[outputs.view(Batch_size)>0.5] = 1
        
        all_labels_b = torch.cat((all_labels_b,labels_b.cuda()),0)
        all_labels_d = torch.cat((all_labels_d,labels_d.cuda()),0)

        all_outputs_pred = torch.cat((all_outputs_pred,outputs_pred),0)

        current_correct += (outputs_pred == target).sum().item()
        correct_positives += ((outputs_pred == target) & (target == 1)).sum().item()
        correct_negatives += ((outputs_pred == target) & (target == 0)).sum().item()
        positives += (target == 1).sum().item()
        negatives += (target == 0).sum().item()
        total += target.size()[0]
        
    accuracy = current_correct/total
    pos_accuracy = correct_positives/positives
    neg_accuracy = correct_negatives/negatives 
        
#     print('Time Since Start: %s' % (timeSince(start)))
    return accuracy, pos_accuracy, neg_accuracy, all_labels_b, all_labels_d, all_outputs_pred

def train_and_test(Train_loader, Test_loader, Model, Criterion, Optimizer, Seq_length, Epochs):
    
     # Start timer for feedback purposes
    start = time.time()
    print('Time Since Start: %s' % (timeSince(start)))
    
    Loss_readout, Acc_readout = [],[]
    all_acc_train, all_acc_test = [],[]
    loss_current, accuracy_current = 0,0
    bin_size = 10
    
    for i in range(Epochs):
        for j, (data, labels_b, labels_d) in enumerate(Train_loader):
            # initialize inputs and target labels
            inputs, target = data.cuda(), labels_b.cuda()
            # initialize context state as an input to network
            c_state = Model.init_c_state().cuda()
            # Zero the gradients of the network
            Model.zero_grad()
            # Run model to generate outputs and new c_state.
            # Run model for as many times it takes to complete a sequence
            input_size = Model.input_size
            for k in range(Seq_length):
                inputs_seq = inputs[:,input_size*k:input_size*(k+1)]
                outputs, c_state = Model(inputs_seq, c_state)
            # Line needed when doing BCELoss in binary classification
            target_f = target.view(target.size()[0],1).float() 
            #Determine loss and optimize network
            loss = Criterion(outputs, target_f)
            loss.backward()
            Optimizer.step()
            
            loss_current += loss.item()
            
            outputs_pred = torch.zeros(inputs.size()[0]).long().cuda()
            outputs_pred[outputs.view(inputs.size()[0])>0.5] = 1
            
            accuracy_current += (outputs_pred==target).sum().item()/inputs.size()[0]
            
            if j % bin_size == 0:
                Loss_readout.append(loss_current/bin_size)
                loss_current = 0
                Acc_readout.append(accuracy_current/bin_size)
                accuracy_current = 0
        
        accuracy_train = Acc_readout[len(Acc_readout)-1]
        
        accuracy_test, posacc, negacc = test(Test_loader, Model, Seq_length)
        
        all_acc_train.append(accuracy_train)
        all_acc_test.append(accuracy_test)
    
    # Start timer for feedback purposes
    print('Time Since Start: %s' % (timeSince(start)))
    
    return all_acc_train, all_acc_test

def train_Elman(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs=5, T=3, t0=0, t1=3):
    '''
    Training Loop for Model.
    Inputs: Train_loader, Model, Criterion, Optimizer, Epochs
    Outputs: Accuracy_array, Loss_array
    '''
    # Initialize records
    acc_array, loss_array = [],[]
    acc_curr, loss_curr = 0,0
    
    ave_valid_acc_array = []
    
    #Initialize time profile for inputs
    t_prof = np.zeros(T)
    t_prof[t0:t1] = 1
    print(t_prof)
    
    # Start Epochs
    for epoch in range(Epochs):
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().cuda()
            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            # Zero the gradients
            Model.zero_grad()
            
            # Forward Step
            # Adding Timesteps
            for j in range(T):
                if t_prof[j] == 0:
                    # Create blank image
                    blank_images = torch.zeros_like(images).cuda()
                    outputs, hidden = Model(blank_images, hidden)
                else:
                    outputs, hidden = Model(images, hidden)
            
            # Backward Step
            loss = Criterion(torch.squeeze(outputs), labels)
            loss.backward()
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = torch.zeros(images.size()[0]).cuda()
            predicted[torch.squeeze(outputs)>0.5] = 1
            
            acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            acc_array.append(acc_curr)
            
            loss_array.append(loss.item())
            
        # Validation accuracy calculation
        valid_acc_array = []
        valid_acc_curr = 0
        
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().cuda()
            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            # Zero the gradients
            Model.zero_grad()
            # Forward Step
            # Adding Timesteps
            for j in range(T):
                if t_prof[j] == 0:
                    # Create blank image
                    blank_images = torch.zeros_like(images).cuda()
                    outputs, hidden = Model(blank_images, hidden)
                else:
                    outputs, hidden = Model(images, hidden)

            
            # Calculate Accuracy
            predicted = outputs.round()
            
            valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_acc_array.append(valid_acc_curr)
        ave_valid_acc = np.mean(valid_acc_array)
        ave_valid_acc_array.append(ave_valid_acc)
        print('Epoch: %d Loss: %.5f Train Acc: %.5f Valid Acc %.5f' % (epoch, loss.item(), acc_curr, ave_valid_acc))
    return(model, acc_array, loss_array, ave_valid_acc_array)

def test_Elman(Test_loader, Model, T=3, t0=0, t1=3):
    '''
    Testing loop for trained Model.
    Inputs: Test_loader, Model
    Outputs: Test Accuracy
    '''
    # Accuracy calculation
    acc_array = []
    acc_curr = 0
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda()
    
    #Initialize time profile for inputs
    t_prof = np.zeros(T)
    t_prof[t0:t1] = 1
    print(t_prof)

    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().cuda()
        
        # Initialize Hidden Layer
        hidden = Model.init_Hidden(images.size()[0]).cuda()
        # Zero the gradients
        Model.zero_grad()
        # Forward Step
        # Adding Timesteps
        for j in range(T):
            if t_prof[j] == 0:
                # Create blank image
                blank_images = torch.zeros_like(images).cuda()
                outputs, hidden = Model(blank_images, hidden)
            else:
                outputs, hidden = Model(images, hidden)

        # Calculate Accuracy
        predicted = outputs.round()
        
        acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]

        acc_array.append(acc_curr)
        
        
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        
        
    ave_acc = np.mean(acc_array)
    return(ave_acc, all_pred, all_l_b, all_l_d)

def train_test(Train_loader, Valid_loader, Test_loader, Model, Criterion, Optimizer, Epochs=5):
    '''
    Training Loop for Model.
    Inputs: Train_loader, Model, Criterion, Optimizer, Epochs
    Outputs: Accuracy_array, Loss_array
    '''
    # Initialize records
    acc_array, loss_array = [],[]
    acc_curr, loss_curr = 0,0
    
    ave_valid_acc_array = []
    ave_test_acc_array = []
    
    #Set up timesteps based on architecture:
    if Model.tree:
        T = int(Model.depth+1)
    else:
        T = int(Model.comps)
    
    print('T=', T)
    
    # Start Epochs
    for epoch in range(Epochs):
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().cuda()
            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Forward Step
            # Adding Timesteps
            for j in range(T):
                outputs, hidden = Model(images, hidden)
            
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step(Model.freeze_mask, Model.soma_mask)
            
            # Calculate Accuracy
            predicted = outputs.round()
            
            labels = labels.long()
            acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            acc_array.append(acc_curr)
            
            loss_array.append(loss.item())
            
        # Validation accuracy calculation
        valid_acc_array = []
        valid_acc_curr = 0
        
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.cuda()
            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            # Zero the gradients
            Model.zero_grad()
            # Forward Step
            # Adding Timesteps
            for j in range(T):
                outputs, hidden = Model(images, hidden)
            
            # Calculate Accuracy
            predicted = outputs.round()

    
            valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]

            valid_acc_array.append(valid_acc_curr)
        ave_valid_acc = np.mean(valid_acc_array)
        ave_valid_acc_array.append(ave_valid_acc)
        
        # Test accuracy calculation
        test_acc_array = []
        test_acc_curr = 0
        
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Test_loader):
            images, labels = images.cuda(), labels_b.cuda()
            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            # Zero the gradients
            Model.zero_grad()
            # Forward Step
            # Adding Timesteps
            for j in range(T):
                outputs, hidden = Model(images, hidden)
            
            # Calculate Accuracy
            predicted = outputs.round()

            test_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]

            test_acc_array.append(test_acc_curr)
        ave_test_acc = np.mean(test_acc_array)
        ave_test_acc_array.append(ave_test_acc)
        print('Epoch: %d Loss: %.5f Train Acc: %.5f Valid Acc %.5f Test Acc %.5f' % (epoch, loss.item(), acc_curr, ave_valid_acc, ave_test_acc))
    return(Model, acc_array, loss_array, ave_valid_acc_array, ave_test_acc_array, hidden, outputs)

# For Neuron_Capacity.ipynb
def train_2L(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, Record=50):
    '''
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size
    
    if Model.Arch == 'rnn':
        T = 3
    else:
        T = 1

    # Start epochs
    for epoch in range(Epochs):
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
            # Initialize Hidden layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()
            
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Forward Step
            for j in range(T):
                outputs, hidden = Model(images, hidden)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/i)
        ave_loss_array.append(loss_curr/i)
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
            # Initialize Hidden Layer
            hidden = Model.init_Hidden(images.size()[0]).cuda()

            # Zero the gradients
            Model.zero_grad()

            # Forward Step
            for j in range(T):
                outputs, hidden = Model(images, hidden)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/i)
        ave_valid_loss_array.append(valid_loss_curr/i)
        
        valid_acc_curr, valid_loss_curr = 0,0

                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array

def test_2L(Test_loader, Model, Criterion):
    acc_array, loss_array = [],[]
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda()
    
    if Model.Arch == 'rnn':
        T = 3
    else:
        T = 1    
        
    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        hidden = Model.init_Hidden(images.size()[0]).cuda()
        for j in range(T):
            outputs, hidden = Model(images, hidden)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_pred, all_l_b, all_l_d)

def train_FCNN(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, ES=True, Patience=20, Record=50):
    '''
    Train loop for FCNN
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size

    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Forward Step
            outputs = Model(images)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/i+.00000000001)
        ave_loss_array.append(loss_curr/i+.00000000001)
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Forward Step
            outputs = Model(images)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0

                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch

def train_FCNN_xor(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, ES=True, Patience=20, Record=50):
    '''
    Train loop for FCNN
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size

    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Forward Step
            outputs = Model(images)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/i+.00000000001)
        ave_loss_array.append(loss_curr/i+.00000000001)
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Forward Step
            outputs = Model(images)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0

                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_FCNN(Test_loader, Model, Criterion):
    acc_array, loss_array = [],[]
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda() 
    all_images = torch.Tensor().cuda()
    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        outputs = Model(images)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b, all_l_d)

def test_FCNN_xor(Test_loader, Model, Criterion):
    acc_array, loss_array = [],[]
    
    all_l_b, all_pred = torch.FloatTensor().cuda(), torch.FloatTensor().cuda() 
    all_images = torch.Tensor().cuda()
    Model.eval()
    for i, (images, labels_b) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        outputs = Model(images)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_pred = torch.cat((all_pred, predicted),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b)


def train_SRNN(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, Timesteps=4, ES=True, Patience=20, Record=50):
    '''
    Troubleshooting 3. early stopping.
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size
    
    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            # Freeze select weights by zeroing out gradients
            for child in Model.children():
                for param in child.parameters():
                    if param.grad.shape == Model.freeze_mask.shape:
                        param.grad[Model.freeze_mask] = 0
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/(i+.00000000001))
        ave_loss_array.append(loss_curr/(i+.00000000001))
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0
                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_SRNN(Test_loader, Model, Criterion, Timesteps=4):
    acc_array, loss_array = [],[]
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda() 
    all_images = torch.Tensor().cuda()
    
    bsize = Test_loader.batch_size

    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        
        # Initialize hidden/comp layer values
        hidden = Model.init_Hidden(images.size()[0]).cuda()

        # Forward Step
        for i in range(Timesteps): 
            outputs, hidden = Model(images, hidden)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b, all_l_d)

def train_SRNN_xor(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, Timesteps=4, ES=True, Patience=20, Record=50):
    '''
    Troubleshooting 3. early stopping.
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size
    
    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            # Freeze select weights by zeroing out gradients
            for child in Model.children():
                for param in child.parameters():
                    if param.grad.shape == Model.freeze_mask.shape:
                        param.grad[Model.freeze_mask] = 0
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/(i+.00000000001))
        ave_loss_array.append(loss_curr/(i+.00000000001))
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0
                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_SRNN_xor(Test_loader, Model, Criterion, Timesteps=4):
    acc_array, loss_array = [],[]
    
    all_l_b, all_pred = torch.FloatTensor().cuda(), torch.FloatTensor().cuda() 
    all_images = torch.Tensor().cuda()
    
    bsize = Test_loader.batch_size

    Model.eval()
    for i, (images, labels_b) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        
        # Initialize hidden/comp layer values
        hidden = Model.init_Hidden(images.size()[0]).cuda()

        # Forward Step
        for i in range(Timesteps): 
            outputs, hidden = Model(images, hidden)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_pred = torch.cat((all_pred, predicted),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b)

def train_M2LNN(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, ES=True, Patience=20, Record=50):
    '''
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size

    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Forward Step
            outputs = Model(images)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            # Freeze select weights by zeroing out gradients
            for child in Model.children():
                for param in child.parameters():
                    if param.grad.shape == Model.freeze_mask_i2h.shape:
                        param.grad[Model.freeze_mask_i2h] = 0
                    if param.grad.shape == Model.freeze_mask_h2n.shape:
                        param.grad[Model.freeze_mask_h2n] = 0
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/i+.00000000001)
        ave_loss_array.append(loss_curr/i+.00000000001)
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Forward Step
            outputs = Model(images)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/i+.00000000001)
        ave_valid_loss_array.append(valid_loss_curr/i+.00000000001)
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/i+.00000000001, Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0

                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_M2LNN(Test_loader, Model, Criterion):
    acc_array, loss_array = [],[]
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda() 
    all_images = torch.Tensor().cuda()
    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        outputs = Model(images)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b, all_l_d)


def train_BRNN_xor(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, Timesteps=4, ES=True, Patience=20, Record=50):
    '''
    Troubleshooting 3. early stopping.
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size
    
    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            # Freeze select weights by zeroing out gradients
            for child in Model.children():
                for param in child.parameters():
                    if param.grad.shape == Model.freeze_mask.shape:
                        param.grad[Model.freeze_mask] = 0
                    if param.grad.shape == Model.freeze_mask_2.shape:
                        param.grad[Model.freeze_mask_2] = 0
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/(i+.00000000001))
        ave_loss_array.append(loss_curr/(i+.00000000001))
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0
                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_BRNN_xor(Test_loader, Model, Criterion, Timesteps=4):
    acc_array, loss_array = [],[]
    
    all_l_b, all_pred = torch.FloatTensor().cuda(), torch.FloatTensor().cuda() 
    all_images = torch.Tensor().cuda()
    
    bsize = Test_loader.batch_size

    Model.eval()
    for i, (images, labels_b) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        
        # Initialize hidden/comp layer values
        hidden = Model.init_Hidden(images.size()[0]).cuda()

        # Forward Step
        for i in range(Timesteps): 
            outputs, hidden = Model(images, hidden)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_pred = torch.cat((all_pred, predicted),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b)

def train_BRNN(Train_loader, Valid_loader, Model, Criterion, Optimizer, Epochs, Timesteps=4, ES=True, Patience=20, Record=50):
    '''
    Troubleshooting 3. early stopping.
    Structure: initialize records, begin epochs, read out data, run model,
    calculate loss, loss backward, optimizer step, calculate accuracy and record accuracy and loss.
    Then run same stuff on valid_loader data.
    '''
    # Initialize records
    ave_acc_array, ave_loss_array = [],[]
    acc_curr, loss_curr = 0,0
    ave_valid_acc_array, ave_valid_loss_array = [],[]
    valid_acc_curr, valid_loss_curr = 0,0
    
    bsize = Train_loader.batch_size
    
    ### Adding earlystopping tool
    if ES:
        early_stopping = EarlyStopping(patience=Patience, verbose=True)
        
    # Start epochs
    for epoch in range(Epochs):
        # Report final epoch
        final_epoch = epoch
        Model.train()
        for i, (images, labels_b, labels_d) in enumerate(Train_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break
            if images.size()[0] < bsize:
                print('somethings wrong')
                
            # Zero the gradients
            Optimizer.zero_grad()
            
            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)
                
            # Backward Step
            loss = Criterion(outputs, labels)
            loss.backward()
            
            # Freeze select weights by zeroing out gradients
            for child in Model.children():
                for param in child.parameters():
                    if param.grad.shape == Model.freeze_mask.shape:
                        param.grad[Model.freeze_mask] = 0
                    if param.grad.shape == Model.freeze_mask_2.shape:
                        param.grad[Model.freeze_mask_2] = 0
            
            Optimizer.step()
            
            # Calculate Accuracy
            predicted = outputs.round()
            labels = labels.long()

            acc_curr += (predicted==labels.view_as(predicted).float()).sum().item()/predicted.size()[0]
            loss_curr += loss.item()

            # Record Accuracy and Loss
        ave_acc_array.append(acc_curr/(i+.00000000001))
        ave_loss_array.append(loss_curr/(i+.00000000001))
        
        acc_curr, loss_curr = 0,0
        
        #Validation
        Model.eval()
        for i, (images, labels_b, labels_d) in enumerate(Valid_loader):
            images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
            if images.size()[0] < bsize:
                break            
                
            # Zero the gradients
            Model.zero_grad()

            # Initialize hidden/comp layer values
            hidden = Model.init_Hidden(images.size()[0]).cuda()
                
            # Forward Step
            for i in range(Timesteps): 
                outputs, hidden = Model(images, hidden)

            loss = Criterion(outputs, labels)

            predicted = outputs.round()
            labels = labels.long()
            valid_acc_curr += predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
            valid_loss_curr += loss.item()
        # Record Accuracy and Loss
        ave_valid_acc_array.append(valid_acc_curr/(i+.00000000001))
        ave_valid_loss_array.append(valid_loss_curr/(i+.00000000001))
        
        #early stopping
        if ES:
            early_stopping(valid_loss_curr/(i+.00000000001), Model)

            if early_stopping.early_stop:
                print('Early Stopping')
                break
        
        valid_acc_curr, valid_loss_curr = 0,0
                
    return Model, ave_acc_array, ave_loss_array, ave_valid_acc_array, ave_valid_loss_array, final_epoch


def test_BRNN(Test_loader, Model, Criterion, Timesteps=4):
    acc_array, loss_array = [],[]
    
    all_l_b, all_l_d, all_pred = torch.LongTensor().cuda(), torch.LongTensor().cuda(), torch.LongTensor().cuda() 
    all_images = torch.Tensor().cuda()
    
    bsize = Test_loader.batch_size

    Model.eval()
    for i, (images, labels_b, labels_d) in enumerate(Test_loader):
        images, labels = images.cuda(), labels_b.float().view(-1,1).cuda()
        
        # Initialize hidden/comp layer values
        hidden = Model.init_Hidden(images.size()[0]).cuda()

        # Forward Step
        for i in range(Timesteps): 
            outputs, hidden = Model(images, hidden)
        loss = Criterion(outputs, labels)
        predicted = outputs.round()
        labels = labels.long()
        valid_acc_curr = predicted.eq(labels.view_as(predicted).float()).sum().item()/images.size()[0]
        valid_loss_curr = loss.item()
        # Record all outcomes
        all_images = torch.cat((all_images, images),0)
        all_l_b = torch.cat((all_l_b, labels_b.cuda()),0)
        all_l_d = torch.cat((all_l_d, labels_d.cuda()),0)
        all_pred = torch.cat((all_pred, predicted.long()),0)
        # Record Accuracy and Loss
        acc_array.append(valid_acc_curr)
        loss_array.append(valid_loss_curr)
    ave_acc = np.mean(acc_array)
    ave_loss = np.mean(loss_array)

    return(ave_acc, ave_loss, all_images, all_pred, all_l_b, all_l_d)

# Make modified training loop so that zeroed weights stay frozen

def train_test_model_sparse(model, trainloader, testloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_curve = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
####        # Freeze select weights by zeroing out gradients
            for child in model.children():
                for param in child.parameters():
                    if param.grad.shape == model.freeze_mask.shape:
                        param.grad[model.freeze_mask] = 0
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 80 == 79:    # print every 80 mini-batches
                loss_curve.append(running_loss/80)
                running_loss = 0.0

    print('Finished Training, %d epochs' % (epoch+1))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct/total
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * accuracy))
        
    return(loss_curve, accuracy, model)

def train_test_ktree(model, trainloader, testloader, epochs=10, randorder=False):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_curve = []
    acc_curve = []
    
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0

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

def train_test_fc(model, trainloader, testloader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_curve = []
    acc_curve = []
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_acc = 0.0

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