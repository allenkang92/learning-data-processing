
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
print(a)
# [1 2 3 4 5 6]

print(a[0])
# 1

a[0] = 10
print(a)
# [10  2  3  4  5  6]

print(a[:2 + 1])
# [10  2  3]

b = a[3:]
print(b)
# [4 5 6]

b[0] = 40
print(a)
# [10  2  3 40  5  6]

