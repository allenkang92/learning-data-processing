


import numpy as np

x = np.arange(0, 9 + 1, 2)
y = np.arange(5)
m = np.vstack([x, y])
print(m)
# [[0 2 4 6 8]
#  [0 1 2 3 4]]

xy = np.hstack([x, y])
print(xy)
# [0 2 4 6 8 0 1 2 3 4]