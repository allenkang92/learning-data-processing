# 8. Reverse a vector (first element becomes last)


import numpy as np

arr = np.arange(5)
print(arr)
# [0 1 2 3 4]

reversed_arr = np.flip(arr)
print(reversed_arr)
# [4 3 2 1 0]


# 다른 방법이 무엇이 있나 확인해봤다.
# reversed_arr = arr[::-1]  # 슬라이싱을 사용한 방법
# 또는
# reversed_arr = np.flipud(arr)  # flipud 함수 사용