

import numpy as np

a = np.arange(12) ** 2 # the first 12 square numbers
print(a)
# [  0   1   4   9  16  25  36  49  64  81 100 121]

i = np.array([1, 1, 3, 8, 5]) # an array of indices
print(i)
# [1 1 3 8 5]

# 팬시 인덱싱
print(a[i]) # the elements of 'a' at the positions 'i'
# [ 1  1  9 64 25]

j = np.array([[3, 4], [9, 7]]) # a bidimensional array of indices
print(a[j]) # the same shape as 'j'
# [[ 9 16]
#  [81 49]]

