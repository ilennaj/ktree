# Make 'classes' file
# TODO: Reassign file save destinations, comment, and remove extra lines of code

import numpy

paired_test = np.load('./results/20200509/confused_pairs_all.npy')

print(paired_test.shape)

ds = 0
pt_mean = np.mean(paired_test, axis=1)

ds_set = ['mnist', 'fmnist', 'kmnist','svhn','usps', 'cifar10']

classes = []

paired_test
for i, ds in enumerate(ds_set):
    focus = pt_mean[i]

    a = np.min(focus[np.nonzero(focus)])
    b = np.sort(focus[np.nonzero(focus)])
    e = b[len(b)-2]
    c = np.where(focus == a)
    d = np.where(focus == e)

    classes.append([c[0][0], c[1][0], ds, a, d[0][0], d[1][0], e])

classes = np.array(classes, dtype=object)
classes_orig = classes
print(classes)
# np.save('./results/20200511/classes.npy', classes[:,:3], allow_pickle=True)

# np.load('./results/20200511/classes.npy', allow_pickle=True)

paired_test = np.load('./results/20200509/confused_pairs_emnist_upper.npy')
paired_test.shape

pt_mean = np.mean(paired_test, axis=1)

ds_set = ['emnist']
classes = []
for i, ds in enumerate(ds_set):
    focus = pt_mean[i]

    a = np.min(focus[np.nonzero(focus)])
    print(a)
    b = np.sort(focus[np.nonzero(focus)])
    e = b[len(b)-2]
    c = np.where(focus == a)
    d = np.where(focus == e)
    
    classes.append([c[0][0]+10, c[1][0]+10, ds, a, d[0][0]+10, d[1][0]+10, e ])

classes = np.array(classes, dtype=object)

print(classes)
# intermed = np.concatenate((class_orig, classes), 0)
# np.save('./results/20200511/classes_emnist_perf.npy', intermed, allow_pickle=True)
# np.save('./results/20200511/classes.npy', classes, allow_pickle=True)

# classes_orig = np.load('./results/20200511/classes.npy', allow_pickle=True)
print(classes_orig.shape, classes[:,:].shape)
classes_final = np.concatenate((classes_orig[:,:3],classes[:,:3]),0)
a = classes_final[3:6].copy()
b = classes_final[6].copy()
classes_final[3] = b
classes_final[4:] = a

print(classes_final)
# # np.save('./results/20200511/classes.npy', classes_final, allow_pickle=True)