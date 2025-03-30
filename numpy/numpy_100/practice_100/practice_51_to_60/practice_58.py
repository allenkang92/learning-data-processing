# 58. Subtract the mean of each row of a matrix

import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr - arr.mean(axis=1, keepdims=True))               
# [[-1.  0.  1.]
#  [-1.  0.  1.]]