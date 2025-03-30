# 61. Find the nearest value from a given value in an array

import numpy as np
import random

arr = np.array([1, 2, 3, 4, 5])
N = random.randint(1, 10)

print(arr[abs(arr - N).argmin()])

