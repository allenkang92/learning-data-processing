

from numpy import newaxis
import numpy as np

# print(np.column_stack((a, b))) # with 2D arrays

a = np.array([4., 2.])
b = np.array([3., 8.])
print(np.column_stack((a, b))) # returns a 2D array
# [[4. 3.]
#  [2. 8.]]

print(np.hstack((a, b))) # the result is different
# [4. 2. 3. 8.]

print(a[:, newaxis]) # view 'a' as a 2D column vector
# [[4.]
#  [2.]]

print(np.column_stack((a[:, newaxis], b[: newaxis])))
# [[4. 3.]
#  [2. 8.]]

print(np.hstack((a[:, newaxis], b[:, newaxis]))) # the result is the same
# [[4. 3.]
#  [2. 8.]]


