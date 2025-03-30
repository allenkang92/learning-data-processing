# 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? 

import numpy as np

arr1 = np.array([1, 2, 3]).reshape((1, 3))
print(arr1)
# [[1 2 3]]

arr2 = np.array([1, 2, 3]).reshape((3, 1))
print(arr2)
# [[1]  
#  [2]
#  [3]]

result = np.add(arr1, arr2)
print(result)
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]