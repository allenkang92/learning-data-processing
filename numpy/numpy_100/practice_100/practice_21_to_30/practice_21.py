# 21. Create a checkerboard 8x8 matrix using the tile function

import numpy as np

arr1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])
arr2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
arr3 = np.vstack((arr1, arr2))


arr3 = np.tile(arr3, (4, 1))

print(arr3)
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]