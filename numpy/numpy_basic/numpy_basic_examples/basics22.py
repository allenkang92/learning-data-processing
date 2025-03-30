

import numpy as np

a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

unique_values = np.unique(a)
print(unique_values)
# [11 12 13 14 15 16 17 18 19 20]


unique_values, indices_list = np.unique(a, return_index = True)
print(indices_list)
# [ 0  2  3  4  5  6  7 12 13 14]

unique_values, occurrence_count = np.unique(a, return_counts = True)
print(occurrence_count)
# [3 2 2 2 1 1 1 1 1 1]

a_2d = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [1, 2, 3, 4]])


unique_values = np.unique(a_2d)
print(unique_values)   
# [ 1  2  3  4  5  6  7  8  9 10 11 12]


unique_rows = np.unique(a_2d, axis = 0)
print(unique_rows)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

unique_rows, indices, occurrence_count = np.unique(
     a_2d, axis=0, return_counts=True, return_index=True)
print(unique_rows)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

print(indices)
# [0 1 2]

print(occurrence_count)
# [2 1 1]