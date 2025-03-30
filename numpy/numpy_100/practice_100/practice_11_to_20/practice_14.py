# 14. Create a random vector of size 30 and find the mean value


import numpy as np

rng = np.random.default_rng()
arr = rng.random((30))

print(arr.mean())
# 0.4206630713273253