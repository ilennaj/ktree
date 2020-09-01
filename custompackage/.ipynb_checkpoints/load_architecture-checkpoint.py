import torch
import torch.nn as nn
import numpy as np
import math


def kronecker(matrix1, matrix2):
    return torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(*(matrix1.size() + matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0) * matrix2.size(0), matrix1.size(1) * matrix2.size(1))


    
class simple_fcnn(nn.Module):
    '''
    2 layer feed forward neural network. 
    Will use leaky ReLU activation functions.
    Activation = {'relu', 'linear'}
    '''
    
    def __init__(self, Input_size=3072, Hidden_size=3072, Output_size=1, Activation="relu"):
        super(simple_fcnn, self).__init__()
        '''
        Inputs: Input_size, Hidden_size, Output_size, Activation
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
        Inputs: Input
        Outputs: Output
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
    k-Tree neural network
    '''
    
    def __init__(self, ds='mnist', Activation="relu", Sparse=True,
                 Input_order=None, Repeats=1, Padded=False):
        super(ktree_gen, self).__init__()
        '''
        Inputs: ds (dataset), activation, sparse, input_order, repeats, padded
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
        # If using 28x28 datasets...
        if (ds == 'mnist') or (ds == 'fmnist') or (ds == 'kmnist') or (ds == 'emnist'):
            # If padded, use 1024 sized tree, completely binary tree
            if Padded:
                self.k = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
            # If not padded, use 784 sized tree, 
            # 7:1 between layers 1 and 2, and layers 2 and 3
            else:
                self.k = [784, 112, 16, 8, 4, 2, 1]
        # If using 3x32x32 datasets...
        elif (ds == 'svhn') or (ds == 'cifar10'):
            # Use 3072 sized tree
            # 3:1 between layers 1 and 2, otherwise binary
            self.k = [3072, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        # If using 16x16 datasets...
        elif ds == 'usps':
            # Use 256 sized tree
            self.k = [256, 128, 64, 32, 16, 8, 4, 2, 1]
        else:
            print('Select a dataset')
            return(None)
        
        # Make layers of tree architecture
        
        # Name each layer in each subtree for reference later
        self.names = np.empty((self.Repeats, len(self.k)-1),dtype=object)
        # Initialize freeze mask for use in training loop
        self.freeze_mask_set = []
        # For each repeat or subtree, make a sparse layer that is initialized correctly
        for j in range(self.Repeats):
            # For each layer within each subtree
            for i in range(len(self.k)-1):
                # Assign name of the layer, indexed by layer (i) and subtree (j)
                name = ''.join(['w',str(j),'_',str(i)])
                # Initialize the layer with the appropriate name
                self.add_module(name, nn.Linear(self.k[i],self.k[i+1]))
                # Set bias of layer to zeros
                self._modules[name].bias = nn.Parameter(torch.zeros_like(self._modules[name].bias)) 
                # Use custom method to re-initialize the layer weights and create freeze mask for that layer
                self._modules[name].weight.data, freeze_mask = self.initialize(self._modules[name])
                # Add the layer name to the list of names
                self.names[j,i] = name
                # Set the freeze mask for the first subtree, which should be the same for all subtrees
                if j < 1:
                    self.freeze_mask_set.append(freeze_mask)
        
        # Initialize root node, aka soma node aka output node
        self.root = nn.Linear(Repeats, 1)
        
        # Initialize nonlinearities
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        '''
        Forward step for network. Establishes Architecture.
        Inputs: Input
        Outputs: Output
        '''
        
        y_out = []
        # Step through every layer in each subtree of model, applying nonlinearities
        for j in range(self.Repeats):
            y = x
            for i in range(len(self.k)-1):
                if self.Activation == 'relu':
                    y = self.relu(self._modules[self.names[j,i]](y))
                else:
                    y = self._modules[self.names[j,i]](y)
            # keep track of pen-ultimate layer outputs
            y_out.append(y)
        
        # Calculate final output, joining the outputs of each subtree together
        output = self.sigmoid(self.root(torch.cat((y_out), dim=1)))

        return(output)
    
    def initialize(self, layer):
        # Kaiming initialize weights accounting for sparsity
        
        # Extract weights from layer we are reinitializing
        weights = layer.weight.data
        # If sparse, change the initializations based on density (sparsity)
        if self.Sparse:
            if weights.shape[1] == 3072: # first layer of 3x32x32 image datasets
                inp_block = torch.ones((1,3))
            elif (weights.shape[1] == 784) or (weights.shape[1] == 112): # first or second layer of 28x28 datasets
                inp_block = torch.ones((1,7))
            else:
                inp_block = torch.ones((1,2)) # all other layers (or 32x32)
            
            # Set up mask for where each node receives a set of inputs of equal size to the input block
            inp_mask = kronecker(torch.eye(weights.shape[0]), inp_block)

            # Calculate density
            density = len(np.where(inp_mask)[0])/len(inp_mask.reshape(-1))

            # Generate Kaiming initialization with gain = 1/density
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1]*density)))
            
            # Where no inputs will be received, set weights to zero
            weights[inp_mask == 0] = 0
        else: # If not sparse, use typical kaiming normalization
            weights = torch.nn.init.normal_(weights, mean=0.0, std=math.sqrt(2/(weights.shape[1])))
        
        # Generate freeze mask for use in training to keep weights initialized to zero at zero
        mask_gen = torch.zeros_like(weights)
        # Indicate where weights are equal to zero
        freeze_mask = mask_gen == weights
        
        return(weights, freeze_mask)

