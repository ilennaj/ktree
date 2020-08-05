import torch
import torch.nn as nn
import numpy as np
import math


def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


    
class simple_fcnn(nn.Module):
    '''
    2 layer feed forward neural network. 
    Will code in Linear, Sigmoid, or ReLU activation functions.
    Activation = {'relu', 'sigmoid', 'linear'}
    '''
    
    def __init__(self, Input_size=3072, Hidden_size=3072, Output_size=1, Activation="relu"):
        super(simple_fcnn, self).__init__()
        '''
        Inputs: Input_size, Hidden_size, Output_size
        '''
        # Initialize architecture parameters
        self.Input_size = Input_size
        self.Hidden_size = Hidden_size
        self.Output_size = Output_size
        self.Activation = Activation
        
        
        # Initialize weights through He initialization (by default in nn.Linear)
        
        self.i2h = nn.Linear(Input_size, Hidden_size, bias=True)
        self.i2h.bias = torch.nn.Parameter(torch.zeros_like(self.i2h.bias))
#         self.i2h.weight = torch.nn.init.normal_(self.i2h.weight, mean=0.0, std=math.sqrt(2/(Input_size)))
        self.i2h.weight = torch.nn.init.kaiming_normal_(self.i2h.weight, a=0.01)


        # Initialize densly connected output layer
        self.h2o = nn.Linear(Hidden_size, Output_size)
        self.h2o.bias = torch.nn.Parameter(torch.zeros_like(self.h2o.bias))
        self.h2o.weight = torch.nn.init.kaiming_normal_(self.h2o.weight, a=0.01)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input, Hidden
        Outputs: Output, Hidden
        '''
        # Prepare input for appropriate architecture

        
        # Set Activation function to calculate hidden layer

        if self.Activation == 'relu':
            Hidden = self.relu(self.i2h(x))
        else:
            Hidden = self.i2h(x)

        # Calculate Output layer
        Output = self.sigmoid(self.h2o(Hidden))
        return(Output)
    
class ktree_gen(nn.Module):
    '''
    Tree NN
    '''
    
    def __init__(self, ds='mnist', Activation="relu", Sparse=True,
                 Input_order=None, Repeats=1, Padded=False):
        super(ktree_gen, self).__init__()
        '''
        Inputs: Input_size, Hidden_size, Output_size
        '''
        # Initialize architecture parameters
        self.ds = ds
        self.Activation = Activation
        self.Sparse = Sparse
        self.Input_order = Input_order
        self.Repeats = Repeats
        
        # Initialize weights
        # Set biases to 0
        # Set kaiming initialize weights with gain to correct for sparsity
        # Set freeze masks
        
        #Specify tree dimensions
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        elif (ds == 'svhn') or (ds == 'cifar10'):
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        elif ds == 'usps':
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        self.freeze_mask_set = []
        for j in range(self.Repeats):
            for i in range(len(self.k)-1):
                name = ''.join(['w',str(j),'_',str(i)])
                self.add_module(name, nn.Linear(self.k[i],self.k[i+1]))
                self._modules[name].bias = nn.Parameter(torch.zeros_like(self._modules[name].bias)) 
                self._modules[name].weight.data, freeze_mask = self.initialize(self._modules[name])
                self.names[j,i] = name
                if j < 1:
                    self.freeze_mask_set.append(freeze_mask)
        
        self.root = nn.Linear(Repeats, 1)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input, Hidden
        Outputs: Output, Hidden
        '''
        
        y_out = []
        for j in range(self.Repeats):
            y = x
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                else:
                    y = self._modules[self.names[j,i]](y)
            y_out.append(y)

        output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))

        return(output)
    
    def initialize(self, layer):
        # Kaiming initialize weights accounting for sparsity
        weights = layer.weight.data
        if self.Sparse:
            if weights.shape[1] == 3072:
                inp_block = torch.ones((1,3))
            elif (weights.shape[1] == 784) or (weights.shape[1] == 112):
                inp_block = torch.ones((1,7))
            else:
                inp_block = torch.ones((1,2))
            inp_mask = kronecker(torch.eye(weights.shape[0]), inp_block)

            density = len(np.where(inp_mask)[0])/len(inp_mask.reshape(-1))

            # Generate Kaiming initialization with gain = 1/density
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1]*density)))

            weights[inp_mask == 0] = 0
        else:
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1])))
        
        mask_gen = torch.zeros_like(weights)
        freeze_mask = mask_gen == weights
        
        return(weights, freeze_mask)

