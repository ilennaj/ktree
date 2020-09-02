# Run Order: 2nd, 1 out of 1
# Make 'classes' file in results folder

import numpy


########
#### Look at confused pairs for all datasets except EMNIST
########
paired_test = np.load('./results/confused_pairs_all.npy')

# Use the mean of each paired class set over all trials
pt_mean = np.mean(paired_test, axis=1)

# Initialize dataset set
ds_set = ['mnist', 'fmnist', 'kmnist','svhn','usps', 'cifar10']

# Initialize classes variable for record keeping
classes = []

# For each dataset
for i, ds in enumerate(ds_set):
    # Select the paired class means for the selected dataset
    focus = pt_mean[i]
    
    # Select pair of classes that have the lowest score
    a = np.min(focus[np.nonzero(focus)])
    c = np.where(focus == a)

    # Record keeping
    classes.append([c[0][0], c[1][0], ds])

classes = np.array(classes, dtype=object)
classes_orig = classes


########
#### Look at confused pairs for only EMNIST
########

paired_test = np.load('./results/confused_pairs_emnist_upper.npy')
paired_test.shape

pt_mean = np.mean(paired_test, axis=1)

# Initialize dataset set
ds_set = ['emnist']

# Initialize classes variable for record keeping
classes = []
# For each dataset (only EMNIST)
for i, ds in enumerate(ds_set):
    # Select the paired class means for the selected dataset
    focus = pt_mean[i]
    
    # Select pair of classes that have the lowest score
    a = np.min(focus[np.nonzero(focus)])
    c = np.where(focus == a)

    # Record keeping
    classes.append([c[0][0]+10, c[1][0]+10, ds])

classes = np.array(classes, dtype=object)

########
#### Organize final class pairs into an array for further use
########

classes_final = np.concatenate((classes_orig,classes),0)
a = classes_final[3:6].copy()
b = classes_final[6].copy()
classes_final[3] = b
classes_final[4:] = a

print(classes_final)
np.save('./results/classes.npy', classes_final, allow_pickle=True)