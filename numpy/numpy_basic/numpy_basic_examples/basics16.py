

import numpy as np

data = np.array([[1, 2],
                 [5, 3],
                 [4, 6]])

print(data)
# [[1 2]
#  [5 3]
#  [4 6]]

print(data.max(axis = 0))
# [5 6]

print(data.max(axis = 1))
# [2 5 6]