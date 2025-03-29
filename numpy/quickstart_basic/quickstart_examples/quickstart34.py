

import numpy as np

a = np.arange(4 + 1)
print(a)
# [0 1 2 3 4]

a[[0, 0, 2]] += 1

print(a)
# [1 1 3 3 4]