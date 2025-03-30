# 30. How to find common values between two arrays? (★☆☆)

import numpy as np

arr1 = np.array([1, 2])
arr2 = np.array([2, 3])

print(np.intersect1d(arr1, arr2))
# [2]