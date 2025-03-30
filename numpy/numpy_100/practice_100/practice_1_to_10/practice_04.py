# 4. How to find the memory size of any array

import numpy as np

arr = np.array([1,2,3,4,5])

print(arr.size)
# 5

# 메모리 사이즈를 확인하지 못했다.
# print(arr.nbytes)  # 바이트 단위의 메모리 크기
# print(arr.itemsize)  # 각 요소의 크기(바이트)
# print(arr.size * arr.itemsize)  # 전체 메모리 크기 계산