# 54. How to read the following file?

# numpy/numpy_100/practice_100/practice_51_to_60/arr_exam.txt
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11

import numpy as np

arr = np.genfromtxt("numpy/numpy_100/practice_100/practice_51_to_60/arr_exam.txt", delimiter=",", dtype=int)
print(arr)
# [[ 1  2  3  4  5]
#  [ 6 -1 -1  7  8]
#  [-1 -1  9 10 11]]