

import numpy as np

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])

print(np.sort(arr))
# [1 2 3 4 5 6 7 8]

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(np.concatenate((a, b)))
# [1 2 3 4 5 6 7 8]


x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])

print(np.concatenate((x, y), axis=0))
# [[1 2]
#  [3 4]
#  [5 6]]
