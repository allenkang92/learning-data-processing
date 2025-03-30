# 37. Create a 5x5 matrix with row values ranging from 0 to 4

import numpy as np

arr = np.array([0, 1, 2, 3, 4])
print(arr)
# [0 1 2 3 4]


arr = np.vstack([arr for _ in range(5)])

print(arr)



# Z = np.zeros((5,5))
# Z += np.arange(5)
# print(Z)

# # without broadcasting
# Z = np.tile(np.arange(0, 5), (5,1))
# print(Z)