


import numpy as np

A = np.array([[1, 1], 
              [0, 1]])

B = np.array([[2, 0], 
              [3, 4]])


print(A * B) # elementwise product
# [[2 0]
#  [0 4]]

print(A @ B) # matrix product
# [[5 4]
#  [3 4]]

print(A.dot(B)) # another matrix product
# [[5 4]
#  [3 4]]