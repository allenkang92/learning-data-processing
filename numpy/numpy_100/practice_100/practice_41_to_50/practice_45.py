# 45. Create random vector of size 10 and replace the maximum value by 0

import numpy as np

arr = np.random.randint(0, 10, 10)

print(arr)
# [7 2 3 3 9 2 9 9 2 4]
print(arr.max())
# 9

arr[arr == arr.max()] = 0
print(arr)
# [7 2 3 3 0 2 0 0 2 4]
