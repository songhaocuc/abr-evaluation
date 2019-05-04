from scipy.io import loadmat
import os
import numpy

filepath = os.path.dirname(__file__) + '/randomForest.mat'
m = loadmat(filepath)
forest = m['forest']
# print(type(numpy.zeros(1)))
print(type(forest))
print(len(forest))
print(type(forest[0]))
atree = forest[0]
print(atree[0])
print(atree[0][0,1])
# for cur_tree in forest:
#     print(type(cur_tree))
#     print(cur_tree)
# print(m['forest'])