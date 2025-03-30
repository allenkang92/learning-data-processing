

import numpy as np

a = np.arange(5 + 1)
print(a)
# [0 1 2 3 4 5]

b = a.reshape(3, 2)
print(b)
# [[0 1]
#  [2 3]
#  [4 5]]

# print(np.reshape(a, shape=(1, 6), order='C'))
# line 15, in <module>
#     print(np.reshape(a, shape=(1, 6), order='C'))
# TypeError: reshape() got an unexpected keyword argument 'shape'