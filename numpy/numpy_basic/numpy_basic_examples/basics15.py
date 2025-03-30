
import numpy as np

data = np.array([[1, 2], [3, 4], [5, 6]])
print(data)
# [[1 2]
#  [3 4]
#  [5 6]]

print(data[0, 1])
# 2

print(data[1:3])
# [[3 4]
#  [5 6]]

print(data[0:2, 0])
# [1 3]

print(data.max())
# 6

print(data.min())
# 1

print(data.sum())
# 21

