# ktree

This github repository is the code for the paper: 

### "Can single neurons solve MNIST? The computational power of biological dendritic trees"

#### Abstract:
Physiological experiments have highlighted how the dendrites of biological neurons can nonlinearly process distributed synaptic inputs. This is in stark contrast to units in artificial neural networks that are generally linear apart from an output nonlinearity. If dendritic trees can be nonlinear, biological neurons may have far more computational power than their artificial counterparts. Here we use a simple model where the dendrite is implemented as a sequence of thresholded linear units. We find that such dendrites can readily solve machine learning problems, such as MNIST or CIFAR-10, and that they benefit from having the same input onto several branches of the dendritic tree. This dendrite model is a special case of sparse network. This work suggests that popular neuron models may severely underestimate the computational power enabled by the biological fact of nonlinear dendrites and multiple synapses per pair of neurons. The next generation of artificial neural networks may significantly benefit from these biologically inspired dendritic architectures.

Preprint can be found here: http://arxiv.org/abs/2009.01269

### Run Order:
1. confused_pairs*
2. combine_classes.py
3. lda.py, fcnn.py
4. ktree_orig.py
5. ktree_perm.py, ktree_rand.py
6. Figures.ipynb


Early Stopping code from this repository: https://github.com/Bjarten/early-stopping-pytorch
