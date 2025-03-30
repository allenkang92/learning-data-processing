# 27. Consider an integer vector Z, which of these expressions are legal?

import numpy as np

Z = np.array([1, 2, 3, 4, 5])
print(Z)
# [1 2 3 4 5]

# Z**Z
# -> [   1    4   27  256 3125]
print(Z**Z)
# [   1    4   27  256 3125]

# 2 << Z >> 2
# -> 모르겠다. 공배열 나올 듯.
print(2 << Z >> 2)
# [ 1  2  4  8 16]

# Z <- Z
# -> 모르겠다.
print(Z <- Z)
# [False False False False False]

# 1j*Z
# -> 복소수
print(1j * Z)
# [0.+1.j 0.+2.j 0.+3.j 0.+4.j 0.+5.j]

# Z/1/1
# 그대로 반환될 듯하다.
print(Z / 1 / 1)
# [1. 2. 3. 4. 5.]

# Z<Z>Z
# -> 공배열 나올 듯하다.
print(Z < Z > Z)
# line 34, in <module>
#     print(Z<Z>Z)
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
