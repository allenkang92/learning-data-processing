# 9. Create a 3x3 matrix with values ranging from 0 to 8

import numpy as np

arr = np.arange(0, 8 + 1)
print(arr)
# [0 1 2 3 4 5 6 7 8]

reshaped_arr = arr.reshape(3, 3)
print(reshaped_arr)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]