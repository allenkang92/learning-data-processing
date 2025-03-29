

import numpy as np

rg = np.random.default_rng(1)

a = np.floor(10 * rg.random((2, 2)))
print(a)
# [[5. 9.]
#  [1. 9.]]

b = np.floor(10 * rg.random((2, 2)))
print(b)
# [[3. 4.]
#  [8. 4.]]

print(np.vstack((a, b)))
# [[5. 9.]
#  [1. 9.]
#  [3. 4.]
#  [8. 4.]]

print(np.hstack((a, b)))
# [[5. 9. 3. 4.]
#  [1. 9. 8. 4.]]