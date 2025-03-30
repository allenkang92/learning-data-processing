
import numpy as np

rng = np.random.default_rng()

print(rng.integers(5, size = (2, 4)))
# [[4 1 1 0]
#  [4 4 4 2]]