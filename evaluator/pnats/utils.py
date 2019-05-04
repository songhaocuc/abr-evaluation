import numpy as np


def range_limit(arr, maximum, minimum):
    # 对一维向量元素的取值范围进行限制
    for i in range(len(arr)):
        arr[i] = max(arr[i], minimum)
        arr[i] = min(arr[i], maximum)
    return arr


if __name__ == '__main__':
    a = np.random.randint(5, size=(4,1))
    b = np.random.randint(5, size=(4,1))
    print(a)
    range_limit(a, 2,1)
    print(a)
    # print(range_limit(a+b, 1, 2) )

