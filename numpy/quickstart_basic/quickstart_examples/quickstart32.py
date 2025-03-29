

import numpy as np

a = np.arange(5)
print(a)
# [0 1 2 3 4]

a[[1, 3, 4]] = 0
print(a)
# [0 0 2 0 0]