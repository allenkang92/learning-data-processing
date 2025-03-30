# 72. How to swap two rows of an array? 

import numpy as np

arr = np.arange(24).reshape(4, 6)
print(arr)
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]

store1 = arr[0]
store2 = arr[1]
arr[0] = store2
arr[1] = store1
print(arr)

# 개선사항:
# 1. 현재 방법은 in-place가 아님 (임시 변수에 저장 후 할당함)
# 2. NumPy의 고급 인덱싱을 사용한 더 효율적인 in-place 스왑 방법:
#    arr[[0, 1]] = arr[[1, 0]]  # 고급 인덱싱으로 한 번에 스왑
# 3. 임시 변수 없이 튜플 언패킹을 사용한 파이썬 방식:
#    arr[0], arr[1] = arr[1].copy(), arr[0].copy()  # copy() 필요 (파이썬 방식)
# 4. 또는 XOR 스왑 알고리즘을 사용한 in-place 방법:
#    단, 이 방법은 NumPy 배열에는 적용 안됨
# 5. 효율성 측면에서는 arr[[0, 1]] = arr[[1, 0]] 방식이 가장 빠름
