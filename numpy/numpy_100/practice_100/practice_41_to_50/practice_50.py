# 50. How to find the closest value (to a given scalar) in a vector? 

import numpy as np

arr = np.array([1, 2, 3, 4, 5])
scalar = 2.71828

print(arr[abs(arr - scalar).argmin()])
# 3