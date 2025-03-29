

import numpy as np

a = np.arange(11 + 1).reshape(3, 4)
print(a)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

i = np.array([[0, 1],
              [1, 2]]) # indices for the first dim of 'a'

j = np.array([[2, 1],
              [3, 3]]) # indices for the second dim

print(a[i, j]) # i and j must have equla shape
# [[ 2  5]
#  [ 7 11]]

print(a[i, 2])
# [[ 2  6]
#  [ 6 10]]

print(a[:, j])
# [[[ 2  1]
#   [ 3  3]]

#  [[ 6  5]
#   [ 7  7]]

#  [[10  9]
#   [11 11]]]

l = (i, j)
# equivalent to a[i, j]
print(a[l])
# [[ 2  5]
#  [ 7 11]]