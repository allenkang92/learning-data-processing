# 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal

import numpy as np

arr = np.zeros((5, 5))
print(arr)
# [[0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]

arr[1, 0] = 1
print(arr)
# [[0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]

arr[2, 1] = 2
print(arr)
# [[0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 2. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0.]]

arr[3, 2] = 3
print(arr)
# [[0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 2. 0. 0. 0.]
#  [0. 0. 3. 0. 0.]
#  [0. 0. 0. 0. 0.]]

arr[4, 3] = 4
print(arr)
# [[0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0.]
#  [0. 2. 0. 0. 0.]
#  [0. 0. 3. 0. 0.]
#  [0. 0. 0. 4. 0.]]


# np.diag(1+np.arange(4),k=-1)