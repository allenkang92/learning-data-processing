# 16. How to add a border (filled with 0's) around an existing array?

import numpy as np

arr = np.ones((3, 3))
print(arr)
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]


# arr = np.pad(arr, 0)
# print(arr)

# np.pad()는 패딩을 추가하는 함수.
# 매개변수로 array, pad_width, mode, constant_values를 받음.
# pad_width는 패딩할 너비를 지정. -> 1을 넣는다면 모든 면에 1칸씩 패딩을 추가
# mode는 패딩 방식을 지정, 'constant'는 일정한 값으로 패딩한다는 의미.
# constant_values는 패딩에 사용할 일정한 값.. mode = 'constant'일 때 패딩에 사용할 값을 지정.

arr = np.pad(arr, pad_width = 1, mode = 'constant', constant_values = 0)

print(arr)
# [[0. 0. 0. 0. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 1. 1. 1. 0.]
#  [0. 0. 0. 0. 0.]]