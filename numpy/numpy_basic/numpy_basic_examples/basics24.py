

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

reversed_arr = np.flip(arr)

print('Reversed Array: ', reversed_arr)
# Reversed Array:  [8 7 6 5 4 3 2 1]

arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

reversed_arr = np.flip(arr_2d)
print(reversed_arr)
# [[12 11 10  9]
#  [ 8  7  6  5]
#  [ 4  3  2  1]]

reversed_arr_rows = np.flip(arr_2d, axis = 0)
print(reversed_arr_rows)
# [[ 9 10 11 12]
#  [ 5  6  7  8]
#  [ 1  2  3  4]]

reversed_arr_columns = np.flip(arr_2d, axis = 1)
print(reversed_arr_columns)
# [[ 4  3  2  1]
#  [ 8  7  6  5]
#  [12 11 10  9]]

arr_2d[1] = np.flip(arr_2d[1])
print(arr_2d)
# [[ 1  2  3  4]
#  [ 8  7  6  5]
#  [ 9 10 11 12]]

arr_2d[:, 1] = np.flip(arr_2d[:, 1])
print(arr_2d)
# [[ 1 10  3  4]
#  [ 8  7  6  5]
#  [ 9  2 11 12]]