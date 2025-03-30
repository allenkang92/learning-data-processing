# 6. Create a null vector of size 10 but the fifth value which is 1 

import numpy as np

arr = np.zeros(10)

print(arr)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

arr[4] = 1
print(arr)
# [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]