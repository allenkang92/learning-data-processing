# 25. Given a 1D array, negate all elements which are between 3 and 8, in place.

import numpy as np

arr = np.arange(3, 8 + 1)
print(arr)
# [3 4 5 6 7 8]

arr[(3 < arr) & (arr < 8)] *= -1
print(arr)
# [ 3 -4 -5 -6 -7  8]
