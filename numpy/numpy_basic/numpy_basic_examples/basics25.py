

import numpy as np

x = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8], 
              [9, 10, 11, 12]])

print(x.flatten())
# [ 1  2  3  4  5  6  7  8  9 10 11 12]

a1 = x.flatten()
a1[0] = 99
print(x) # Original array
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print(a1) # New array
# [99  2  3  4  5  6  7  8  9 10 11 12]

a2 = x.ravel()
a2[0] = 98
print(x) # Original array
# [[98  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print(a2) # New array
# [98  2  3  4  5  6  7  8  9 10 11 12]
