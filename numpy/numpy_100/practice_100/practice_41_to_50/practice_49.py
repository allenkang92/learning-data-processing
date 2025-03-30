# 49. How to print all the values of an array?

import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
print(*arr1)

arr2 = np.array([[1, 2],
                 [3, 4]])

print(arr2)
# [[1 2]
#  [3 4]]

print(*[_ for _ in arr2.flatten()])
# 1 2 3 4