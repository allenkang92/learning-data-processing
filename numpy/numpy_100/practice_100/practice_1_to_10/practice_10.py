

import numpy as np

arr = np.array([1, 2, 0, 0, 4, 0])

print(arr[arr != 0])
# [1 2 4]

# 다른 방법 확인
# non_zero = np.nonzero(arr)[0]  # 0이 아닌 요소의 인덱스 찾기
# print(arr[non_zero])  # 해당 인덱스의 값 출력