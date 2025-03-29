


import numpy as np

rg = np.random.default_rng(1)

a = np.floor(10 * rg.random((3, 4)))
print(a)
# [[5. 9. 1. 9.]
#  [3. 4. 8. 4.]
#  [5. 0. 7. 5.]]

print(a.shape)
# (3, 4)

print(a.ravel()) # returns the array, flattend
# [5. 9. 1. 9. 3. 4. 8. 4. 5. 0. 7. 5.]

print(a.reshape(6, 2)) # returns the array with a modified shape
# [[5. 9.]
#  [1. 9.]
#  [3. 4.]
#  [8. 4.]
#  [5. 0.]
#  [7. 5.]]

print(a.T) # returns the array, transposed
# [[5. 3. 5.]
#  [9. 4. 0.]
#  [1. 8. 7.]
#  [9. 4. 5.]]

print(a.T.shape)
# (4, 3)

print(a.shape)
# (3, 4)

print(a)
# [[5. 9. 1. 9.]
#  [3. 4. 8. 4.]
#  [5. 0. 7. 5.]]

print(a.resize((2, 6)))
# None

print(a)
# [[5. 9. 1. 9. 3. 4.]
#  [8. 4. 5. 0. 7. 5.]]


# reshape 할 때, 차원을 -1로 지정하면 다른 차원은 자동으로 계산해준다.
print(a.reshape(3, -1))
# [[5. 9. 1. 9.]
#  [3. 4. 8. 4.]
#  [5. 0. 7. 5.]]