import numpy as np


def getRfPred(input, forest):
    # This is an implementation of random forest regressor for p.1203.3
    # here input is a 14x1 feature vector
    # forest is a 3-d tensor where the third dimension is # of trees
    # each row in a tree represents a node in tree, consisting of 5
    # entries:
    # 1. node ID
    # 2. feature ID
    # 3. feature threshold (or MOS)
    # 4. left child node ID,
    # 5. right child node ID
    input = np.array(input)
    nTrees = len(forest)
    predictions = np.zeros(nTrees)

    '''core model'''
    for kkk in range(0, nTrees):
        cur_tree = forest[kkk][0]
        # start at the 0-th node
        cur_node = 0

        while True:
            if cur_tree[cur_node, 1] == -1:
                break

            cur_value = input[int(cur_tree[cur_node, 1])]
            cur_thresh = int(cur_tree[cur_node, 2])
            if cur_value < cur_thresh:
                cur_node = int(cur_tree[cur_node, 3])
            else:
                cur_node = int(cur_tree[cur_node, 4])

        predictions[kkk] = cur_tree[cur_node, 2]

    RF_prediction = sum(predictions)/len(predictions)

    return RF_prediction

if __name__ == '__main__':
    from scipy.io import loadmat
    import os
    testArgs = np.array(range(14))
    filepath = '../../randomForest.mat'
    m = loadmat(filepath)
    forest = m['forest']
    val = getRfPred(testArgs, forest)
    print(val)