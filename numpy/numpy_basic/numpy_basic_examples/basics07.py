

import numpy as np

array_example = np.array([[[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                          [4, 5, 6, 7]]])


print(array_example.ndim)
# 3

print(array_example.shape)
# (3, 2, 4)