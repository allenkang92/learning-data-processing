

import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

b1 = np.array([False, True, True]) # first dim selection
b2 = np.array([True, False, True, False]) # second dim selection

print(a[b1, :]) # selecting rows
# [[ 4  5  6  7]
#  [ 8  9 10 11]]

print(a[b1])
# [[ 4  5  6  7]
#  [ 8  9 10 11]]

print(a[:, b2]) # selecting columns
# [[ 0  2]
#  [ 4  6]
#  [ 8 10]]

print(a[b1, b2])
# [ 4 10]
