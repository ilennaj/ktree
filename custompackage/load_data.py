import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms

def format_data_weighted(Data_set, Target_class_1, Target_class_2, Data_weighting='default', permute=False, padded=True):
    '''
    Change labels so that target class is of value 1 and all other classes are 
    of value 0. Dataset will be a 3 member tuple: data, label_binary, label_default.
    Inputs: Data_set, Target_class_1, Target_class_2, Data_weighting, permute, padded
    Outputs: Data_set_formatted
    '''
    # If Data_weighting == paired, then only the 2 classes, with 1:1 data weighting, are returned labeled 0 and 1
    if Data_weighting == 'paired':
        # Load intire dataset with batch size equal to entire dataset
        Loader = DataLoader(Data_set, batch_size=len(Data_set), shuffle=True)
        for _, (inputs, labels) in enumerate(Loader):
            # data and label_default contain entire dataset
            data = inputs
            label_default = labels

        # Filter out all classes except Target classes
        # Get indices for filter
        selector_1 = np.where(label_default.numpy() == Target_class_1)
        selector_2 = np.where(label_default.numpy() == Target_class_2)
        
        # Filter labels and data (images)
        label_1 = label_default[selector_1]
        data_1 = data[selector_1]
        label_2 = label_default[selector_2]
        data_2 = data[selector_2]

        # combine filtered data and label for each class
        label_pair = torch.cat((label_1, label_2), 0)
        data_pair = torch.cat((data_1, data_2), 0)

        # Assign binary labels to each class
        label_binary = np.where(label_pair.numpy() == Target_class_1, 1, 0)
        label_binary = torch.from_numpy(label_binary).long()
        ## permute with get_permutation function
        if permute:
            if data_pair.shape[1:] == torch.Size([1,28,28]):
                # Pad data to make it 32x32
                padding_f = torch.nn.ZeroPad2d(2)
                data_pair = padding_f(data_pair)
                # make 2d image a 1d array
                data_pair = data_pair.view(len(data_pair),-1)
                # get permutation
                perm_idx = get_permutation(5)
                # Permute data
                data_pair = data_pair[:,perm_idx]
            elif data_pair.shape[1:] == torch.Size([3,32,32]): #for CIFAR10 and SVHN datasets
                # make 2d image a 1d array
                data_pair = data_pair.view(len(data_pair),-1)
                # get permutation
                perm_idx = get_permutation(5)
                perm_idx = np.concatenate((perm_idx,perm_idx,perm_idx),0)
                # permute data
                data_pair = data_pair[:,perm_idx]
            else:
                # make 2d image a 1d array
                data_pair = data_pair.view(len(data_pair),-1)
                # get permutation
                perm_idx = get_permutation(4)
                # permute data
                data_pair = data_pair[:,perm_idx]
        else: # Only if padding, then pad 28x28 to 32x32
            if padded and data_pair.shape[1:] == torch.Size([1,28,28]):
                # Pad data
                padding_f = torch.nn.ZeroPad2d(2)
                data_pair = padding_f(data_pair)  
            # make 2d image a 1d array
            data_pair = data_pair.view(len(data_pair),-1)
        # Put now formatted data and labels into a dataset
        Data_set_formatted = torch.utils.data.TensorDataset(data_pair, label_binary, label_pair)
        
    else: # Keep all classes, only label target class with 1 and all others with 0
        # Load intire dataset with batch size equal to entire dataset
        Loader = DataLoader(Data_set, batch_size=len(Data_set), shuffle=True)
        for _, (inputs, labels) in enumerate(Loader):
            # data and label_default contain entire dataset
            data = inputs
            label_default = labels

        # Assign binary labels to each class
        label_binary = np.where(labels.numpy() == Target_class_1, 1, 0)
        label_binary = torch.from_numpy(label_binary).long()
    ## permute with get_permutation function
        if permute:
            if data.shape[1:] == torch.Size([1,28,28]):
                # Pad data
                padding_f = torch.nn.ZeroPad2d(2)
                data = padding_f(data)
                # make 2d image a 1d array
                data = data.view(len(data),-1)
                # get permutation
                perm_idx = get_permutation(5)
                # Permute data
                data = data[:,perm_idx]
            elif data.shape[1:] == torch.Size([3,32,32]):
                # make 2d image a 1d array
                data = data.view(len(data),-1)
                # get permutation
                perm_idx = get_permutation(5)
                perm_idx = np.concatenate((perm_idx,perm_idx,perm_idx),0)
                # Permute data
                data = data[:,perm_idx]
            else:
                # make 2d image a 1d array
                data = data.view(len(data),-1)
                # get permutation
                perm_idx = get_permutation(4)
                # Permute data
                data = data[:,perm_idx]
        else:
            # make 2d image a 1d array
            data = data.view(len(data),-1)
        # Put now formatted data and labels into a dataset
        Data_set_formatted = torch.utils.data.TensorDataset(data, label_binary, labels)
    
    return Data_set_formatted
    
    
def dataset_weighted_split_all(Batch_size=32, Target_class_1=0, Target_class_2=1,
                                 Data_weighting='default', Split=5, ds='mnist', permute=False,
                                 padded=True):
    '''
    Produces dataset that will be fed into a network model.
    Inputs: Batch_size, Target_class_1, Target_class_2, Data_weighting, Split, dataset, permute, padded
    Outputs: set of Train_loaders, set of Valid_loaders, a single Test_loader
    '''
    transform = transforms.ToTensor()
    
    # Load Datasets
    if ds == 'mnist':
        Train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
        Test_set  = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    elif ds == 'fmnist':
        Train_set = torchvision.datasets.FashionMNIST(root='./fmdata', train=True,
                                           download=True, transform=transform)
        Test_set  = torchvision.datasets.FashionMNIST(root='./fmdata', train=False,
                                          download=True, transform=transform)    
    elif ds == 'cifar10':
        Train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True,
                                           download=True, transform=transform)
        Test_set  = torchvision.datasets.CIFAR10(root='./cifardata', train=False,
                                          download=True, transform=transform)   
    elif ds == 'kmnist':
        Train_set = torchvision.datasets.KMNIST(root='./kmnist', train=True, 
                                                transform=transform , download=True)
        Test_set = torchvision.datasets.KMNIST(root='./kmnist', train=False, 
                                               transform=transform, download=True)
    elif ds == 'emnist':
        # 0-9: numbers
        # 10-35: uppercase letters
        # 36-61: lowercase letters
        Train_set = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True,
                                                transform=transform, download=True)
        Test_set = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False, 
                                               transform=transform, download=True)
    elif ds == 'svhn':
        Train_set = torchvision.datasets.SVHN(root='./data', split='train',
                                              transform=transform, download=True)
        Test_set = torchvision.datasets.SVHN(root='./data', split='test',
                                             transform=transform, download=True)
    elif ds == 'usps':
        Train_set = torchvision.datasets.USPS(root='./data', train=True, 
                                              transform=transform, download=True)
        Test_set = torchvision.datasets.USPS(root='./data', train=False, 
                                             transform=transform, download=True)
    else:
        print('Error: Specify dataset')
        return None

    # Assign Binary Labels to target classes
    Train_set = format_data_weighted(Train_set, Target_class_1, Target_class_2, Data_weighting=Data_weighting, permute=permute, padded=padded)
    Test_set  = format_data_weighted(Test_set, Target_class_1, Target_class_2, Data_weighting=Data_weighting, permute=permute, padded=padded)
    
    # Set length for dataset splitting purposes
    train_len = Train_set.tensors[0].size()[0]
    test_len = Test_set.tensors[0].size()[0]

    # Make validset from training data such that it is equal in size to the test set
    Train_set, Valid_set = torch.utils.data.dataset.random_split(Train_set, (train_len-test_len, test_len))

    # Since random_split sends out a subset and the original dataset is normally used from that.
    # remake datasets so that they no longer depend on the original dataset
    Train_set = torch.utils.data.TensorDataset(Train_set[:][0], Train_set[:][1], Train_set[:][2])
    Valid_set = torch.utils.data.TensorDataset(Valid_set[:][0], Valid_set[:][1], Valid_set[:][2])

    if Data_weighting == 'paired': # paired is 1:1 weighting
        
        # Split Training set and Valid set into multiple dataloaders and return array of dataloaders
        Train_loader_split, Valid_loader_split = [],[]
        for i in range(Split):
            # get size of each split dataset
            spl = int(len(Train_set)/Split)
            # Split dataset
            Train_set_split = torch.utils.data.TensorDataset(Train_set[i*spl:(i+1)*spl][0], 
                                                              Train_set[i*spl:(i+1)*spl][1],
                                                              Train_set[i*spl:(i+1)*spl][2])
            # make dataloader from split section of original dataset
            Train_loader = DataLoader(Train_set_split, batch_size=Batch_size, shuffle=True)
            # Add split dataloader to the list of dataloaders
            Train_loader_split.append(Train_loader)

            # get size of each split dataset
            spl = int(len(Valid_set)/Split)
            # Split dataset
            Valid_set_split = torch.utils.data.TensorDataset(Valid_set[i*spl:(i+1)*spl][0], 
                                                              Valid_set[i*spl:(i+1)*spl][1],
                                                              Valid_set[i*spl:(i+1)*spl][2])
            # make dataloader from split section of original dataset
            Valid_loader = DataLoader(Valid_set_split, batch_size=Batch_size, shuffle=True)
            # Add split dataloader to the list of dataloaders
            Valid_loader_split.append(Valid_loader)
        
        # make dataloader from original test set
        Test_loader = DataLoader(Test_set, batch_size=Batch_size, shuffle=False)

    else: #Default is 1:9 oversampled weighting. Oversample target dataset
        
        # Determine ratios of each class
        trainratio = np.bincount(Train_set.tensors[1])
        validratio = np.bincount(Valid_set.tensors[1])
        testratio = np.bincount(Test_set.tensors[1])

        # Convert ratios to counts of how many samples belong to each class
        train_classcount = trainratio.tolist()
        valid_classcount = validratio.tolist() 
        test_classcount = testratio.tolist()

        # Use counts to calculate original dataweighting of each class
        train_weights = 1./torch.tensor(train_classcount, dtype=torch.float)
        valid_weights = 1./torch.tensor(valid_classcount, dtype=torch.float)
        test_weights = 1./torch.tensor(test_classcount, dtype=torch.float)

        # Select target class to get original weights of that class
        train_sampleweights = train_weights[Train_set.tensors[1]]
        valid_sampleweights = train_weights[Valid_set.tensors[1]]
        test_sampleweights = test_weights[Test_set.tensors[1]]

        # Make samplers for data loader in order to oversample target class
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train_sampleweights, 
                                                                       num_samples=len(train_sampleweights))
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=valid_sampleweights, 
                                                                       num_samples=len(valid_sampleweights))
        test_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=test_sampleweights, 
                                                                       num_samples=len(test_sampleweights))
        
        # Split Training set and Valid set into multiple dataloaders and return array of dataloaders
        Train_loader_split, Valid_loader_split = [],[]
        for i in range(Split):
            # get size of each split dataset
            spl = int(len(Train_set)/Split)
            # Split dataset
            Train_set_split = torch.utils.data.TensorDataset(Train_set[i*spl:(i+1)*spl][0], 
                                                              Train_set[i*spl:(i+1)*spl][1],
                                                              Train_set[i*spl:(i+1)*spl][2])
            # make dataloader from split section of original dataset
            Train_loader = DataLoader(Train_set_split, batch_size=Batch_size, shuffle=True)
            # Add split dataloader to the list of dataloaders
            Train_loader_split.append(Train_loader)
            
            # get size of each split dataset
            spl = int(len(Valid_set)/Split)
            # Split dataset
            Valid_set_split = torch.utils.data.TensorDataset(Valid_set[i*spl:(i+1)*spl][0], 
                                                              Valid_set[i*spl:(i+1)*spl][1],
                                                              Valid_set[i*spl:(i+1)*spl][2])
            # make dataloader from split section of original dataset
            Valid_loader = DataLoader(Valid_set_split, batch_size=Batch_size, shuffle=True)
            # Add split dataloader to the list of dataloaders
            Valid_loader_split.append(Valid_loader)
        # make dataloader from original test set
        Test_loader = DataLoader(Test_set, batch_size=Batch_size, sampler=test_sampler)
        
    return Train_loader_split, Valid_loader_split, Test_loader




def get_matrix(n):
    '''
     Assumes that the matrix is of size 2^n x 2^n for some n
    
     EXAMPLE for n=4
    
     Old order:
    
      1  2  3  4
      5  6  7  8
      9 10 11 12
     13 14 15 16
    
     New order:
    
      1  2  5  6
      3  4  7  8
      9 10 13 14
     11 12 15 16
    
     Function returns numbers from old order, read in the order of the new numbers:
    
     [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]
    
     So if you previously had a data vector v from a matrix size 32 x 32,
     you can now use v[get_permutation(5)] to reorder the elements.
    '''
    if n == 0:
        return np.array([[1]])
    else:
        smaller = get_matrix(n - 1)
        num_in_smaller = 2 ** (2 * n - 2)
        first_stack = np.hstack((smaller, smaller + num_in_smaller))
        return np.vstack((first_stack, first_stack + 2 * num_in_smaller))

def get_permutation(n):
    return get_matrix(n).ravel() - 1