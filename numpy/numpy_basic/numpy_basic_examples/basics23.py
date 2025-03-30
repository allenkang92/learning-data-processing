

import numpy as np

arr = np.arange(6).reshape((2, 3))
print(arr)
# [[0 1 2]
#  [3 4 5]]

print(arr.transpose())
# [[0 3]
#  [1 4]
#  [2 5]]

print(arr.T)
# [[0 3]
#  [1 4]
#  [2 5]]